"""
Open-set LLM family classifier based on class-mean templates.

Unlike the ensemble (closed-set, requires retraining for new families), this:
  - Classifies by cosine distance to the nearest per-family mean fingerprint
  - Adds new families from just a few fingerprint samples (no retraining needed)
  - Provides principled OOD detection via distance ratio + calibrated radius
  - Controls family identification and unknown rejection during identify

Workflow:
  1. Build templates from training data:      tc = TemplateClassifier()
         tc.build(simulation_data)   # dict: family -> [vectors]
         tc.save(path)

  2. Classify at inference:
         tc.load(path)
         result = tc.classify(fingerprint_vector)

  3. Add a new family without retraining:
         tc.add_family("deepseek", new_vectors)
         tc.save(path)
"""

import logging
import numpy as np
import joblib

from llm_fingerprinter import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_distances(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine distance from a single query vector to each row of matrix.

    Returns values in [0, 2]:  0 = identical, 1 = orthogonal, 2 = opposite.
    Uses numpy only (no scipy dependency).
    """
    q = query / (np.linalg.norm(query) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    return 1.0 - np.clip(sims, -1.0, 1.0)


def _fit_standardizer(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature (mean, std) over a (N, D) stack. Constant features get std=1."""
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TemplateClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TemplateClassifier:
    """Open-set LLM family classifier based on class-mean templates."""

    def __init__(self, ood_ratio_threshold: float = 0.80):
        """
        Args:
            ood_ratio_threshold: Flag OOD when
                best_distance / second_best_distance > this value.
                A ratio near 1.0 means the two closest classes are
                equidistant — no clear winner.  Lower = stricter OOD.
                Default 0.80 works well in practice.
        """
        self.templates: dict[str, np.ndarray] = {}   # family -> mean vector
        self.ood_ratio_threshold = ood_ratio_threshold
        self._ood_radius: float | None = None         # calibrated from training
        # Per-feature standardizer, fitted in build().
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self.is_built = False
        # model_name -> family, set by build-model-templates for OOD family recovery.
        self.model_families: dict[str, str] = {}

    # ── Internal: feature standardization ──────────────────────────────────────

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """Standardize X (D,) or (N, D); no-op if no scaler was fitted."""
        if self._feat_mean is None or self._feat_std is None:
            return X
        if (X.ndim not in (1, 2) or self._feat_mean.ndim != 1
                or X.shape[-1] != len(self._feat_mean)
                or self._feat_std.shape != self._feat_mean.shape
                or not np.isfinite(self._feat_mean).all()
                or not np.isfinite(self._feat_std).all()
                or np.any(self._feat_std <= 0)):
            raise ValueError("Fingerprint and template scaler geometry must match exactly")
        transformed = ((X - self._feat_mean) / self._feat_std).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("Nonfinite features after template standardization")
        return transformed

    # ── Build / update ───────────────────────────────────────────────────────

    def build(self, simulation_data: dict,
              model_families: dict = None) -> bool:
        """Compute per-family class-mean template from training fingerprints.

        Args:
            simulation_data: dict mapping family_name -> list[np.ndarray]
            model_families:  Optional dict mapping model_name -> family (str).
                             When provided, classify() can return an
                             inferred_family for the winning model even when
                             the caller only knows the model name, not the
                             family.  Pass the result of
                             FingerprintStore.export_model_family_map().

        Returns:
            True on success.
        """
        if not simulation_data:
            logger.error("No simulation data provided")
            return False

        populated = {k: v for k, v in simulation_data.items() if len(v) > 0}

        if len(populated) < 2:
            logger.error(
                f"Need at least 2 classes to build templates, got "
                f"{len(populated)} ({sorted(populated) or 'none'}). A one-class "
                f"store yields a zero-vector template and meaningless distances. "
                f"Collect fingerprints for another family/model first."
            )
            return False

        self.is_built = False
        self._ood_radius = None
        self.templates = {}
        self.model_families = model_families or {}
        intra_distances: list[float] = []

        # Standardize first — otherwise cosine distance is dominated by the
        # large-magnitude linguistic counts and embeddings contribute ~0%.
        all_vecs = [np.asarray(v, dtype=np.float32)
                    for vs in simulation_data.values() for v in vs]
        if not all_vecs:
            logger.error("No vectors in simulation data — templates not built")
            return False
        stacked = np.stack(all_vecs)
        if stacked.ndim != 2 or not np.isfinite(stacked).all():
            logger.error("Template training requires finite, one-dimensional vectors")
            return False
        self._feat_mean, self._feat_std = _fit_standardizer(stacked)
        # Reference scale for the degeneracy check below.
        typical_norm = float(np.median(
            np.linalg.norm(self._apply_scaler(np.stack(all_vecs)), axis=1)
        ))

        for family, vectors in simulation_data.items():
            if not vectors:
                logger.warning(f"Skipping '{family}': no vectors")
                continue

            vecs = self._apply_scaler(np.array(vectors, dtype=np.float32))
            mean_vec = vecs.mean(axis=0)
            self.templates[family] = mean_vec

            # Collect intra-class distances for OOD radius calibration
            if len(vecs) > 1:
                dists = _cosine_distances(mean_vec, vecs)
                intra_distances.extend(dists.tolist())

        if not self.templates:
            logger.error("No valid families — templates not built")
            return False

        # Guard against templates that collapsed to (near) zero in standardized
        # space — cosine distance to such a vector carries no signal.
        degenerate = [
            f for f, t in self.templates.items()
            if float(np.linalg.norm(t)) < 1e-3 * max(typical_norm, 1e-9)
        ]
        if degenerate:
            logger.error(
                f"Templates collapsed to ~zero in standardized space: {degenerate}. "
                f"Distances would be numerical noise, so the store was not built. "
                f"This usually means the classes are not actually distinct."
            )
            self.templates = {}
            return False

        # OOD radius: 2× the 95th-percentile intra-class distance
        # A test fingerprint further than this from its nearest template
        # is likely from an unknown family.
        if intra_distances:
            self._ood_radius = float(np.percentile(intra_distances, 95)) * 2.0
            logger.info(f"OOD radius calibrated to {self._ood_radius:.4f}")

        self.is_built = True
        logger.info(
            f"Built {len(self.templates)} templates: {sorted(self.templates)}"
        )
        return True

    def add_family(self, family_name: str, vectors: list) -> bool:
        """Add (or overwrite) a single family template without rebuilding others.

        Requires at least 3 fingerprints for a reliable mean.

        Args:
            family_name: Name for the new family (e.g. "deepseek")
            vectors:     List of fingerprint np.ndarray from simulate

        Returns:
            True on success.
        """
        if not vectors:
            logger.error(f"No vectors for '{family_name}'")
            return False
        if len(vectors) < 3:
            logger.warning(
                f"Only {len(vectors)} vector(s) for '{family_name}' — "
                f"recommend >= 3 for a reliable template"
            )

        if self._feat_mean is None:
            logger.warning(
                f"Adding '{family_name}' to a store with no fitted standardizer — "
                f"template stored in raw feature space. Run 'build-templates' to "
                f"refit the standardizer across all families for best accuracy."
            )
        vecs = self._apply_scaler(np.array(vectors, dtype=np.float32))
        if vecs.ndim != 2 or not np.isfinite(vecs).all():
            logger.error("Cannot add non-finite or malformed template vectors")
            return False
        mean_vec = vecs.mean(axis=0)
        typical_norm = float(np.median(np.linalg.norm(vecs, axis=1)))
        if float(np.linalg.norm(mean_vec)) < 1e-3 * max(typical_norm, 1e-9):
            logger.error("Cannot add '%s': template centroid has no usable signal", family_name)
            return False
        self.templates[family_name] = mean_vec

        # Fold this family's spread into the OOD radius. Without this the radius
        # stays calibrated on whatever classes build() saw, so a newly added
        # family sits outside it and every probe of it reads as OOD.
        if len(vecs) > 1:
            new_radius = float(np.percentile(
                _cosine_distances(mean_vec, vecs), 95)) * 2.0
            self._ood_radius = (new_radius if self._ood_radius is None
                                else max(self._ood_radius, new_radius))
            logger.info(f"OOD radius after add_family: {self._ood_radius:.4f} "
                        f"(re-run 'build-templates' to recalibrate across all families)")

        self.is_built = True
        logger.info(
            f"Added template for '{family_name}' from {len(vectors)} vectors "
            f"(total families: {len(self.templates)})"
        )
        return True

    # ── Classify ─────────────────────────────────────────────────────────────

    def classify(self, fingerprint: np.ndarray, top_k: int = 3) -> dict:
        """Classify a fingerprint by nearest template.

        Args:
            fingerprint: Raw 1206-dim feature vector (standardized internally).
            top_k:       Number of ranked candidates to return

        Returns:
            dict with keys:
              family       - predicted family name (or 'unknown' if OOD)
              distance     - cosine distance to nearest template (lower = closer)
              confidence   - 1 - distance/2, scaled to [0,1]
              ranked       - list of {'family', 'distance'} sorted best-first
              is_ood       - True if prediction is uncertain
              ood_reason   - 'ratio' | 'radius' | None
        """
        if not self.is_built:
            raise RuntimeError(
                "Templates not built. Call build() first or load() from disk."
            )

        if not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        fingerprint = np.asarray(fingerprint, dtype=np.float32)
        if fingerprint.ndim != 1 or not np.isfinite(fingerprint).all():
            raise ValueError("Fingerprint must be a finite, one-dimensional vector")
        families = list(self.templates.keys())
        template_matrix = np.stack([self.templates[f] for f in families])
        if (template_matrix.ndim != 2 or not np.isfinite(template_matrix).all()
                or fingerprint.shape != (template_matrix.shape[1],)):
            raise ValueError("Fingerprint width must match finite template vectors exactly")
        fp = self._apply_scaler(fingerprint)

        dists = _cosine_distances(fp, template_matrix)
        order = np.argsort(dists)

        ranked = [
            {"family": families[i], "distance": float(dists[i])}
            for i in order[:top_k]
        ]

        best = ranked[0]
        # Rejection must not depend on the number of displayed candidates.
        second_distance = float(dists[order[1]]) if len(order) > 1 else None

        # OOD signal 1: ratio test — best and second-best are too similar
        ratio_ood = (
            second_distance is not None
            and (best["distance"] / (second_distance + 1e-9))
            > self.ood_ratio_threshold
        )
        # OOD signal 2: absolute distance exceeds calibrated radius
        radius_ood = (
            self._ood_radius is not None
            and best["distance"] > self._ood_radius
        )

        # A store with a single template cannot discriminate at all: the ratio
        # test has no runner-up to compare against and the radius was calibrated
        # from that same lone class, so both signals are inert. Report unknown
        # rather than a confident nearest match.
        lone_template = len(self.templates) < 2
        uncalibrated = self._ood_radius is None or not np.isfinite(self._ood_radius)
        no_signal = np.linalg.norm(fp) <= 1e-12

        is_ood = ratio_ood or radius_ood or lone_template or uncalibrated or no_signal
        if lone_template:
            ood_reason = "insufficient_templates"
        elif uncalibrated:
            ood_reason = "uncalibrated_radius"
        elif no_signal:
            ood_reason = "no_signal"
        else:
            ood_reason = "ratio" if ratio_ood else ("radius" if radius_ood else None)

        # Confidence: invert cosine distance (0 = perfect match → 1.0 confidence)
        confidence = float(max(0.0, 1.0 - best["distance"] / 2.0))

        # If a model_name -> family mapping was provided at build time, look up
        # the family for the winning model so callers can use it as a fallback
        # when the ensemble classifier is OOD but the model template is not.
        inferred_family = self.model_families.get(best["family"]) if not is_ood else None

        return {
            "family": "unknown" if is_ood else best["family"],
            "predicted_family": best["family"],
            "distance": round(best["distance"], 4),
            "confidence": round(confidence, 4),
            "ranked": ranked,
            "is_ood": is_ood,
            "ood_reason": ood_reason,
            "inferred_family": inferred_family,
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, filepath: str) -> bool:
        """Save templates to disk."""
        try:
            joblib.dump(
                {
                    "templates": self.templates,
                    "ood_ratio_threshold": self.ood_ratio_threshold,
                    "ood_radius": self._ood_radius,
                    "is_built": self.is_built,
                    "model_families": self.model_families,
                    "feat_mean": self._feat_mean,
                    "feat_std": self._feat_std,
                },
                filepath,
            )
            logger.info(
                f"Saved {len(self.templates)} templates to {filepath}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Load templates from disk.

        Raises:
            config.UntrustedArtifactError: if `filepath` is outside every
                trusted root (joblib.load unpickles, which executes code).
        """
        filepath = config.ensure_trusted_artifact(filepath)
        try:
            data = joblib.load(filepath)
            self.templates = data["templates"]
            self.ood_ratio_threshold = data.get("ood_ratio_threshold", 0.80)
            self._ood_radius = data.get("ood_radius")
            self.is_built = data.get("is_built", True)
            self.model_families = data.get("model_families", {})
            self._feat_mean = data.get("feat_mean")
            self._feat_std = data.get("feat_std")
            return True
        except config.UntrustedArtifactError:
            raise
        except FileNotFoundError:
            logger.debug(f"Templates file not found: {filepath}")
            return False
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            return False

    # ── Info ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = f"{len(self.templates)} families" if self.is_built else "not built"
        return f"TemplateClassifier({status})"
