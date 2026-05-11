"""
Open-set LLM family classifier based on class-mean templates.

Unlike the ensemble (closed-set, requires retraining for new families), this:
  - Classifies by cosine distance to the nearest per-family mean fingerprint
  - Adds new families from just a few fingerprint samples (no retraining needed)
  - Provides principled OOD detection via distance ratio + calibrated radius
  - Runs alongside the ensemble as a second opinion during identify

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
        self.is_built = False
        # Optional model_name -> family mapping (populated by build-model-templates).
        # When present, classify() returns an inferred_family so that identify()
        # can recover the correct family even when the ensemble is OOD.
        self.model_families: dict[str, str] = {}

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

        self.templates = {}
        self.model_families = model_families or {}
        intra_distances: list[float] = []

        for family, vectors in simulation_data.items():
            if not vectors:
                logger.warning(f"Skipping '{family}': no vectors")
                continue

            vecs = np.array(vectors, dtype=np.float32)
            mean_vec = vecs.mean(axis=0)
            self.templates[family] = mean_vec

            # Collect intra-class distances for OOD radius calibration
            if len(vecs) > 1:
                dists = _cosine_distances(mean_vec, vecs)
                intra_distances.extend(dists.tolist())

        if not self.templates:
            logger.error("No valid families — templates not built")
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

        vecs = np.array(vectors, dtype=np.float32)
        self.templates[family_name] = vecs.mean(axis=0)
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
            fingerprint: Raw 1206-dim feature vector (same as ensemble input)
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

        fp = fingerprint.astype(np.float32).ravel()
        families = list(self.templates.keys())
        template_matrix = np.stack([self.templates[f] for f in families])

        dists = _cosine_distances(fp, template_matrix)
        order = np.argsort(dists)

        ranked = [
            {"family": families[i], "distance": float(dists[i])}
            for i in order[:top_k]
        ]

        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None

        # OOD signal 1: ratio test — best and second-best are too similar
        ratio_ood = (
            second is not None
            and (best["distance"] / (second["distance"] + 1e-9))
            > self.ood_ratio_threshold
        )
        # OOD signal 2: absolute distance exceeds calibrated radius
        radius_ood = (
            self._ood_radius is not None
            and best["distance"] > self._ood_radius
        )

        is_ood = ratio_ood or radius_ood
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
        """Load templates from disk."""
        try:
            data = joblib.load(filepath)
            self.templates = data["templates"]
            self.ood_ratio_threshold = data.get("ood_ratio_threshold", 0.80)
            self._ood_radius = data.get("ood_radius")
            self.is_built = data.get("is_built", True)
            self.model_families = data.get("model_families", {})
            return True
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
