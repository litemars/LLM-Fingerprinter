import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from llm_fingerprinter import config
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

logger = logging.getLogger(__name__)


class EnsembleClassifier:

    def __init__(self, model_families = None,
                 use_pca = False,
                 pca_components = 64,
                 augment_data = True,
                 augment_noise_std = 0.01,
                 augment_samples = 5,
                 embedding_pca_dim = 64,
                 n_layers = 3,
                 ood_confidence_threshold = 0.3,
                 ood_disagreement_threshold = 0.15,
                 early_stop_variants = False,
                 random_state = 42):

        self.model_families = model_families if model_families is not None else config.MODEL_FAMILIES

        self.families_inv = {v: k for k, v in self.model_families.items()}
        self.n_classes = len(self.model_families)

        # PCA settings
        self.use_pca = use_pca
        self.pca_target_components = pca_components
        self.pca_components = None
        self.pca = None

        # Embedding rebalancing settings
        self.embedding_pca_dim = embedding_pca_dim
        self.n_layers = n_layers
        self.embedding_pcas = None
        self.per_layer_block_size = None
        self.per_layer_embed_dim = None
        self.preprocessing_version = 2
        self.feature_mask = None
        self.family_templates = None

        # OOD detection settings
        self.ood_confidence_threshold = ood_confidence_threshold
        self.ood_disagreement_threshold = ood_disagreement_threshold

        self.early_stop_variants = early_stop_variants

        # Augmentation settings
        self.augment_data = augment_data
        self.augment_noise_std = augment_noise_std
        self.augment_samples = augment_samples
        self.random_state = random_state

        # Classifiers
        self.rf = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.svm = SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=random_state,
            class_weight='balanced'
        )
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )

        self.scaler = StandardScaler()
        self.is_trained = False
        self.input_dim = None

        mode = "with PCA" if use_pca else "without PCA"
        logger.info(f"Initialized EnsembleClassifier {mode}, {self.n_classes} classes, "
                    f"embedding rebalancing {embedding_pca_dim}d")

    def _generate_early_stop_variants(self, X: np.ndarray, y: np.ndarray):
        """Add synthetic partial fingerprints matching early-stopped inference.

        Mirrors _build_partial_fingerprint's padding rule (remaining layers =
        mean of completed ones) for each early-stop point, with the same labels,
        so the classifier handles partial fingerprints. Run before augmentation.
        """
        n_features = X.shape[1]
        if n_features % self.n_layers != 0:
            logger.warning(
                f"_generate_early_stop_variants: {n_features} dims not divisible "
                f"by n_layers={self.n_layers} — skipping partial-fingerprint augmentation"
            )
            return X, y

        per_layer = n_features // self.n_layers
        X_parts = [X]
        y_parts = [y]

        for stop_after in range(1, self.n_layers):   # 1-layer done, 2-layers done, …
            # Completed blocks for this early-stop scenario
            completed = [
                X[:, i * per_layer:(i + 1) * per_layer]
                for i in range(stop_after)
            ]
            # fallback: mean across the completed blocks, shape (n_samples, per_layer)
            fallback = np.mean(completed, axis=0)

            X_partial = X.copy()
            for i in range(stop_after, self.n_layers):
                X_partial[:, i * per_layer:(i + 1) * per_layer] = fallback

            X_parts.append(X_partial)
            y_parts.append(y)

        X_out = np.vstack(X_parts)
        y_out = np.concatenate(y_parts)
        logger.info(
            f"Early-stop variants: {len(X)} full → {len(X_out)} total "
            f"({self.n_layers - 1} synthetic partial variant(s) per sample)"
        )
        return X_out, y_out

    def _augment_samples(self, X: np.ndarray, y: np.ndarray):
        """Augment with additive Gaussian noise scaled per-feature std, clipped to
        the observed range. Additive (not multiplicative) so near-zero features
        still get perturbed."""
        if not self.augment_data or len(X) == 0:
            return X, y

        X_min  = X.min(axis=0)
        X_max  = X.max(axis=0)
        X_std  = X.std(axis=0)          # per-feature spread of real training data
        # Fall back to a small absolute value where std is zero (constant feature)
        X_std  = np.where(X_std > 0, X_std, 1e-6)

        X_aug_list = [X]
        y_aug_list = [y]
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.augment_samples):
            noise    = rng.normal(0, self.augment_noise_std, X.shape)
            X_noisy  = X + noise * X_std
            # Clip to observed range to prevent impossible values
            X_noisy  = np.clip(X_noisy, X_min, X_max)
            X_aug_list.append(X_noisy)
            y_aug_list.append(y)

        X_aug = np.vstack(X_aug_list)
        y_aug = np.concatenate(y_aug_list)

        logger.info(f"Augmented {len(X)} → {len(X_aug)} samples")
        return X_aug, y_aug

    @staticmethod
    def _fit_stable_pca(X, max_components, precision):
        """Fit only components supported by the centered observation matrix.

        Use the numerical-rank tolerance for the source precision, even though
        fitting uses float64. Casting float32 inputs must not turn their rounding
        noise into extra measured dimensions. No variance threshold is tuned on
        validation data.
        """
        count = min(int(max_components), len(X) - 1, X.shape[1])
        if count <= 0 or not np.any(np.ptp(X, axis=0)):
            return None
        pca = PCA(n_components=count, svd_solver="full")
        pca.fit(X)
        tolerance = pca.singular_values_[0] * max(X.shape) * precision
        rank = int(np.count_nonzero(pca.singular_values_ > tolerance))
        if rank == 0:
            return None
        
        pca.components_ = pca.components_[:rank]
        pca.explained_variance_ = pca.explained_variance_[:rank]
        pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:rank]
        pca.singular_values_ = pca.singular_values_[:rank]
        pca.n_components_ = rank
        pca.n_components = rank
        return pca

    def _rebalance_features(self, X: np.ndarray, fit: bool = False):
        """Compress embedding dimensions per layer block to rebalance feature groups.

        Input: (n_samples, n_layers * per_layer_dim) e.g. (N, 1206)
        Output: (n_samples, n_layers * (embedding_pca_dim + non_embed_dim)) e.g. (N, 246)
        """
        n_features = X.shape[1]
        if n_features % self.n_layers != 0:
            raise ValueError(
                f"Feature dim {n_features} is not divisible by n_layers "
                f"{self.n_layers} — cannot split it into per-layer blocks"
            )
        non_embed_dim = config.LINGUISTIC_DIM + config.BEHAVIORAL_DIM  # 18

        if fit:
            if not hasattr(self, "_fit_precision"):
                dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.dtype("float64")
                self._fit_precision = np.finfo(dtype).eps
            per_layer_dim = n_features // self.n_layers
            embed_dim = per_layer_dim - non_embed_dim
            if embed_dim <= 0 or self.embedding_pca_dim < 1:
                raise ValueError("Each layer must contain embeddings and embedding_pca_dim must be positive")
            self.per_layer_block_size = per_layer_dim
            self.per_layer_embed_dim = embed_dim
            self.embedding_pcas = []
        else:
            per_layer_dim = (self.per_layer_block_size
                             if self.per_layer_block_size is not None
                             else n_features // self.n_layers)
            embed_dim = (self.per_layer_embed_dim
                         if self.per_layer_embed_dim is not None
                         else per_layer_dim - non_embed_dim)
            expected = per_layer_dim * self.n_layers
            if n_features != expected:
                raise ValueError(
                    f"Fingerprint has {n_features} features but this classifier "
                    f"was fitted on {expected} "
                    f"({self.n_layers} layers x {per_layer_dim}). Re-run 'train', "
                    f"or regenerate the fingerprint with the matching version."
                )

        if self.embedding_pcas is None:
            return X

        rebalanced_blocks = []
        for layer_idx in range(self.n_layers):
            start = layer_idx * per_layer_dim
            embeddings = X[:, start:start + embed_dim]
            non_embeddings = X[:, start + embed_dim:start + per_layer_dim]

            if fit:
                pca = self._fit_stable_pca(
                    embeddings, self.embedding_pca_dim, self._fit_precision)
                self.embedding_pcas.append(pca)
            else:
                pca = self.embedding_pcas[layer_idx]
            # A constant embedding block has rank zero and contributes no
            # measured information; keep its linguistic/behavioral features.
            compressed = (pca.transform(embeddings) if pca is not None
                          else np.empty((len(X), 0)))
            if fit:
                variance = pca.explained_variance_ratio_.sum() if pca is not None else 0.0
                logger.info(f"Layer {layer_idx} embedding PCA: {compressed.shape[1]} components, "
                           f"{variance:.1%} variance retained")

            rebalanced_blocks.append(np.hstack([compressed, non_embeddings]))

        return np.hstack(rebalanced_blocks)

    def _preprocess(self, X: np.ndarray, fit = False):
        X = np.asarray(X)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("Features must be a finite two-dimensional matrix")
        if fit:
            source_dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.dtype("float64")
            self._fit_precision = np.finfo(source_dtype).eps
            X = X.astype(np.float64)
            self.input_dim = X.shape[1]
            self.preprocessing_version = 2

            # Step 1: Rebalance features (compress embeddings per layer)
            X_rebalanced = self._rebalance_features(X, fit=True)
            logger.info(f"Rebalanced: {X.shape[1]} -> {X_rebalanced.shape[1]} dimensions")

            # Step 2: scale measured features, preserving geometry within each
            # PCA embedding block. Independently standardizing every component
            # would whiten it and amplify weak/unstable directions.
            self.scaler.fit(X_rebalanced)
            tolerance = self._fit_precision * np.maximum(
                1.0, np.abs(self.scaler.mean_))
            self.feature_mask = np.sqrt(self.scaler.var_) > tolerance
            offset = 0
            non_embed_dim = self.per_layer_block_size - self.per_layer_embed_dim
            for pca in self.embedding_pcas:
                width = pca.n_components_ if pca is not None else 0
                if width:
                    block = slice(offset, offset + width)
                    # A common RMS scale gives the block unit average variance
                    # without magnifying its low-variance components.
                    self.scaler.scale_[block] = np.sqrt(np.mean(self.scaler.var_[block]))
                    self.feature_mask[block] = True
                offset += width + non_embed_dim
            if not np.any(self.feature_mask):
                raise ValueError("Training data has no varying features")
            X_scaled = self.scaler.transform(X_rebalanced)[:, self.feature_mask]

            # Step 3: Optional global PCA
            if self.use_pca:
                self.pca = self._fit_stable_pca(
                    X_scaled, self.pca_target_components, self._fit_precision)
                if self.pca is None:
                    raise ValueError("Training data has no usable PCA components")
                self.pca_components = self.pca.n_components_
                X_out = self.pca.transform(X_scaled)
                variance = self.pca.explained_variance_ratio_.sum()
                logger.info(f"Global PCA: {X_out.shape[1]} components, {variance:.1%} variance")
            else:
                X_out = X_scaled
                logger.info(f"Using rebalanced features: {X_out.shape[1]} dimensions")
        else:
            if self.input_dim is not None and X.shape[1] != self.input_dim:
                raise ValueError(
                    f"Fingerprint has {X.shape[1]} features but this classifier "
                    f"was trained on {self.input_dim}. This usually means the "
                    f"fingerprint and the classifier were produced by different "
                    f"versions — re-run 'train'."
                )
            X_rebalanced = self._rebalance_features(X, fit=False)
            X_scaled = self.scaler.transform(X_rebalanced)
            if self.feature_mask is not None:
                X_scaled = X_scaled[:, self.feature_mask]
            if self.use_pca and self.pca is not None:
                X_out = self.pca.transform(X_scaled)
            else:
                X_out = X_scaled

        return X_out

    def train(self, X: np.ndarray, y: np.ndarray):
        X, y = np.asarray(X), np.asarray(y)
        self.is_trained = False
        self.family_templates = None
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            logger.error("Training requires a feature matrix and one label per row")
            return False
        if X.shape[0] == 0:
            logger.error("Empty training data")
            return False

        if X.shape[0] < self.n_classes:
            logger.warning(f"Only {X.shape[0]} samples for {self.n_classes} classes")

        logger.info(f"Training on {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"PCA mode: {'enabled' if self.use_pca else 'disabled (raw features)'}")

        try:
            # Learn representation only from genuinely observed fingerprints.
            # Augmentation can regularize the classifiers but cannot add rank
            # or change the scale of the measured training population.
            self._preprocess(X, fit=True)

            # Step 1: optionally add synthetic partial-fingerprint variants so
            # the classifier handles early-stopped inference correctly. Off by
            # default — see the note in __init__.
            if self.early_stop_variants:
                X_exp, y_exp = self._generate_early_stop_variants(X, y)
            else:
                X_exp, y_exp = X, y
                logger.info(
                    "Early-stop variants disabled — training on real fingerprints "
                    "only. Pass --early-stop-variants if you use `identify --early-stop`."
                )

            # Step 2: noise augmentation on full + partial variants
            X_train, y_train = self._augment_samples(X_exp, y_exp)

            # Step 3: Preprocess (rebalance + scale, optionally PCA)
            X_processed = self._preprocess(X_train, fit=False)
            
            logger.info(f"Training features shape: {X_processed.shape}")

            # Train classifiers
            logger.info("Training Random Forest...")
            self.rf.fit(X_processed, y_train)
            
            logger.info("Training SVM...")
            self.svm.fit(X_processed, y_train)
            
            logger.info("Training MLP...")
            # MLP has no class_weight; imbalance is handled by RF/SVM (balanced).
            _, counts = np.unique(y_train, return_counts=True)
            self.mlp.set_params(early_stopping=(
                counts.min() >= 2 and
                int(np.ceil(len(y_train) * self.mlp.validation_fraction)) >= len(counts)
            ))
            self.mlp.fit(X_processed, y_train)

            from llm_fingerprinter.template_classifier import TemplateClassifier
            templates = TemplateClassifier()
            if not templates.build({
                self.families_inv[int(label)]: list(X[y == label])
                for label in np.unique(y)
            }):
                raise ValueError("Could not build family templates from the training observations")
            self.family_templates = templates

            self.is_trained = True
            logger.info("Training complete")
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def predict_with_confidence(self, fingerprint):
        if not self.is_trained:
            logger.error("Classifier not trained")
            return None, 0.0, {}, {}

        try:
            if fingerprint.ndim == 1:
                fingerprint = fingerprint.reshape(1, -1)

            # Preprocess (rebalance + scale, optionally PCA)
            fp_processed = self._preprocess(fingerprint, fit=False)
            logger.debug(f"Processed fingerprint shape: {fp_processed.shape}")

            # Expand each classifier's output into the full n_classes space.
            # sklearn only emits columns for classes seen in training, so a
            # family with no samples (e.g. 'gemini') is missing from classes_;
            # scatter by class_id into a zeroed array to keep labels aligned.
            n = self.n_classes
            rf_pred  = np.zeros(n, dtype=np.float64)
            svm_pred = np.zeros(n, dtype=np.float64)
            mlp_pred = np.zeros(n, dtype=np.float64)

            # predict_proba once each (SVM Platt scaling is costly), then scatter.
            rf_proba  = self.rf.predict_proba(fp_processed)[0]
            svm_proba = self.svm.predict_proba(fp_processed)[0]
            mlp_proba = self.mlp.predict_proba(fp_processed)[0]

            for col, class_id in enumerate(self.rf.classes_):
                rf_pred[class_id]  = rf_proba[col]
            for col, class_id in enumerate(self.svm.classes_):
                svm_pred[class_id] = svm_proba[col]
            for col, class_id in enumerate(self.mlp.classes_):
                mlp_pred[class_id] = mlp_proba[col]

            logger.debug(f"RF prediction (aligned):  {rf_pred}")
            logger.debug(f"SVM prediction (aligned): {svm_pred}")
            logger.debug(f"MLP prediction (aligned): {mlp_pred}")

            # Weighted ensemble
            weights = [0.45, 0.45, 0.10]  # RF, SVM, MLP
            ensemble_pred = (weights[0] * rf_pred +
                           weights[1] * svm_pred +
                           weights[2] * mlp_pred)
            logger.debug(f"Ensemble prediction: {ensemble_pred}")
            top_idx = np.argmax(ensemble_pred)

            if top_idx not in self.families_inv:
                logger.error(f"Predicted class {top_idx} not in family mapping")
                return None, 0.0, {}, {}

            top_family = self.families_inv[top_idx]
            top_confidence = ensemble_pred[top_idx]

            # Build probability dict (only families with training data or > 0)
            probs = {
                self.families_inv[i]: float(ensemble_pred[i])
                for i in range(n)
                if i in self.families_inv
            }

            # OOD detection: check confidence and classifier agreement
            top_per_classifier = [np.argmax(rf_pred), np.argmax(svm_pred), np.argmax(mlp_pred)]
            agreement_ratio = sum(1 for t in top_per_classifier if t == top_idx) / 3.0
            confidence_per_classifier = [rf_pred[top_idx], svm_pred[top_idx], mlp_pred[top_idx]]
            confidence_std = float(np.std(confidence_per_classifier))

            # Threshold 0.6 keeps a 2-of-3 vote (0.667) from counting as
            # disagreement; only a minority agreeing flags OOD.
            is_ood = (top_confidence < self.ood_confidence_threshold or
                      (agreement_ratio < 0.6 and confidence_std > self.ood_disagreement_threshold))

            ood_info = {
                'is_ood': is_ood,
                'agreement_ratio': agreement_ratio,
                'confidence_std': confidence_std,
                'classifier_top_classes': {
                    'rf':  self.families_inv.get(int(top_per_classifier[0]), '?'),
                    'svm': self.families_inv.get(int(top_per_classifier[1]), '?'),
                    'mlp': self.families_inv.get(int(top_per_classifier[2]), '?'),
                },
            }

            if is_ood:
                logger.warning(f"OOD detected: confidence={top_confidence:.3f}, "
                             f"agreement={agreement_ratio:.2f}, std={confidence_std:.3f}")

            return top_family, float(top_confidence), probs, ood_info

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, 0.0, {}, {}

    def save(self, filepath):
        """Save trained classifier to file."""
        try:
            data = {
                'classifier_type': 'ensemble',
                'rf': self.rf,
                'svm': self.svm,
                'mlp': self.mlp,
                'scaler': self.scaler,
                'pca': self.pca,
                'use_pca': self.use_pca,
                'pca_components': self.pca_components,
                'pca_target_components': self.pca_target_components,
                'model_families': self.model_families,
                'is_trained': self.is_trained,
                'input_dim': self.input_dim,
                'augment_data': self.augment_data,
                'augment_noise_std': self.augment_noise_std,
                'augment_samples': self.augment_samples,
                'early_stop_variants': self.early_stop_variants,
                'random_state': self.random_state,
                'preprocessing_version': self.preprocessing_version,
                'feature_mask': self.feature_mask,
                'family_templates': self.family_templates,
                # Embedding rebalancing
                'embedding_pcas': self.embedding_pcas,
                'embedding_pca_dim': self.embedding_pca_dim,
                'n_layers': self.n_layers,
                'per_layer_block_size': self.per_layer_block_size,
                'per_layer_embed_dim': self.per_layer_embed_dim,
                # OOD detection
                'ood_confidence_threshold': self.ood_confidence_threshold,
                'ood_disagreement_threshold': self.ood_disagreement_threshold,
            }
            joblib.dump(data, filepath)
            mode = "with PCA" if self.use_pca else "rebalanced features"
            logger.info(f"Saved classifier ({mode}) to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save classifier: {e}")
            return False

    def load(self, filepath):
        """Load trained classifier from file.

        Raises:
            config.UntrustedArtifactError: if `filepath` is outside every
                trusted root. joblib.load unpickles, which executes code from
                whoever wrote the file, so this is checked before opening it
                and is deliberately NOT swallowed by the except below.
        """
        filepath = config.ensure_trusted_artifact(filepath)
        try:
            data = joblib.load(filepath)

            self.rf = data['rf']
            self.svm = data['svm']
            self.mlp = data['mlp']
            self.scaler = data['scaler']
            self.pca = data.get('pca')
            self.use_pca = data.get('use_pca', False)
            self.pca_components = data.get('pca_components', 64)
            self.pca_target_components = data.get('pca_target_components', 64)
            self.model_families = data['model_families']
            self.is_trained = data.get('is_trained', True)
            self.input_dim = data.get('input_dim')
            self.augment_data = data.get('augment_data', True)
            self.augment_noise_std = data.get('augment_noise_std', 0.01)
            self.augment_samples = data.get('augment_samples', 5)
            self.early_stop_variants = data.get('early_stop_variants', True)
            self.random_state = data.get('random_state', 42)
            self.preprocessing_version = data.get('preprocessing_version', 1)
            self.feature_mask = data.get('feature_mask')
            self.family_templates = data.get('family_templates')

            # Embedding rebalancing
            self.embedding_pcas = data.get('embedding_pcas')
            self.embedding_pca_dim = data.get('embedding_pca_dim', 64)
            self.n_layers = data.get('n_layers', 3)
            self.per_layer_block_size = data.get('per_layer_block_size')
            self.per_layer_embed_dim = data.get('per_layer_embed_dim')

            # OOD detection
            self.ood_confidence_threshold = data.get('ood_confidence_threshold', 0.3)
            self.ood_disagreement_threshold = data.get('ood_disagreement_threshold', 0.15)

            self.families_inv = {v: k for k, v in self.model_families.items()}
            self.n_classes = len(self.model_families)

            mode = "with PCA" if self.use_pca else "rebalanced features"
            logger.info(f"Loaded classifier ({mode}) from {filepath}")
            return True
        except config.UntrustedArtifactError:
            raise
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return False


    def train_from_simulations(self, simulation_data):
        if len(simulation_data) == 0:
            logger.error("No simulation data provided")
            return False

        X_list, y_list = [], []
        family_counts = {}

        for family_name, vectors in simulation_data.items():
            if family_name not in self.model_families:
                logger.warning(f"Unknown family: {family_name}")
                continue
            if len(vectors) == 0:
                continue

            class_id = self.model_families[family_name]
            family_counts[family_name] = len(vectors)

            for vector in vectors:
                if not isinstance(vector, np.ndarray):
                    vector = np.array(vector, dtype=np.float32)
                X_list.append(vector)
                y_list.append(class_id)

        if len(X_list) == 0:
            logger.error("No valid training samples")
            return False

        # Check dimension consistency
        dims = [v.shape[0] for v in X_list]
        if len(set(dims)) > 1:
            logger.error(f"Inconsistent feature dimensions: {set(dims)}")
            return False

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)

        # Surface families defined in MODEL_FAMILIES but with no training data
        # (handled at inference, but worth flagging).
        missing = [f for f in self.model_families if f not in family_counts]
        if missing:
            logger.warning(
                f"No training data for: {missing}. "
                f"These families will score 0% at inference — "
                f"run 'simulate --family <name>' to add them."
            )

        logger.info(f"Training from simulations: {family_counts}")
        return self.train(X, y)


    @staticmethod
    def _grouped_splits(y, groups, n_folds, random_state):
        """Assign whole model groups within each family to preserve coverage.

        Generic stratified group heuristics can place a scarce family's models
        in the same fold. Here every family contributes at least one complete
        model group to every validation fold and remains present in training.
        """
        y, groups = np.asarray(y), np.asarray(groups)
        if groups.ndim != 1 or groups.shape != y.shape:
            raise ValueError("groups must contain one model label per training row")
        for group in np.unique(groups):
            if len(np.unique(y[groups == group])) != 1:
                raise ValueError(f"Model group {group!r} has conflicting family labels")
        rng = np.random.default_rng(random_state)
        folds = [[] for _ in range(n_folds)]
        total_sizes = np.zeros(n_folds, dtype=int)
        for label in np.unique(y):
            class_groups = np.unique(groups[y == label])
            if len(class_groups) < n_folds:
                raise ValueError("Every family needs at least one model group per fold")
            members = [np.flatnonzero(groups == group) for group in class_groups]
            # Shuffle ties, then greedily distribute large groups first.
            order = rng.permutation(len(members))
            order = sorted(order, key=lambda i: -len(members[i]))
            class_sizes = np.zeros(n_folds, dtype=int)
            for idx in order:
                candidates = np.flatnonzero(class_sizes == class_sizes.min())
                candidates = candidates[total_sizes[candidates] == total_sizes[candidates].min()]
                fold = int(rng.choice(candidates))
                folds[fold].extend(members[idx].tolist())
                class_sizes[fold] += len(members[idx])
                total_sizes[fold] += len(members[idx])
        all_indices = np.arange(len(y))
        present = set(np.unique(y))
        splits = []
        for fold in folds:
            val_idx = np.array(sorted(fold), dtype=int)
            train_idx = np.setdiff1d(all_indices, val_idx)
            if set(y[train_idx]) != present or set(y[val_idx]) != present:
                raise ValueError("Grouped split omitted a family from training or validation")
            splits.append((train_idx, val_idx))
        return splits

    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5,
                       groups=None, random_state: int = 42):
        """Run k-fold cross-validation and return per-family metrics.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Label vector (n_samples,).
            n_folds: Number of folds (default 5).
            groups: Optional per-sample group label (the model each fingerprint
                came from). Whole model groups are assigned within each family,
                keeping models disjoint and every known family on both sides.

        Returns:
            Dict with keys:
            - fold_accuracies: list of per-fold accuracy
            - mean_accuracy: float
            - per_family: dict mapping family name -> {precision, recall, f1}
            - confusion_matrix: np.ndarray
            - grouped: whether group-aware splitting was used
            - limiting_family / n_groups: what capped the fold count
            - argmax_mean_accuracy / argmax_fold_accuracies: closed-set ensemble
              scores, before the shared identification pipeline's rejection
            - accepted_coverage / accepted_accuracy: final-policy coverage and
              accuracy conditional on an accepted family
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        from llm_fingerprinter.identification import IdentificationPipeline

        X, y = np.asarray(X), np.asarray(y)

        # Fold count from the smallest non-empty class — reserved/empty families
        # show up as a 0 count and would otherwise force folds to 0.
        class_counts = np.bincount(y.astype(int))
        present_counts = class_counts[class_counts > 0]
        smallest_class = int(present_counts.min()) if present_counts.size else 0

        grouped = groups is not None
        limiting_family = None
        n_groups = None

        if grouped:
            groups = np.asarray(groups)
            if groups.ndim != 1 or groups.shape != y.shape:
                raise ValueError("groups must contain one model label per training row")
            # A class can be split at most as many ways as it has distinct models.
            per_class_groups = {
                int(c): len(set(groups[y == c])) for c in np.unique(y)
            }
            # Sort by (group count, family name) so ties report deterministically.
            limiting_class = min(
                per_class_groups,
                key=lambda c: (per_class_groups[c], self.families_inv.get(c, str(c)))
            )
            n_groups = per_class_groups[limiting_class]
            limiting_family = self.families_inv.get(limiting_class, str(limiting_class))
            actual_folds = min(n_folds, n_groups, smallest_class)
        else:
            logger.warning(
                "Cross-validating without group labels: repeated simulations of "
                "the same model will be split across folds, so the accuracy "
                "reported here is optimistic."
            )
            actual_folds = min(n_folds, smallest_class)

        if actual_folds < 2:
            if grouped:
                logger.warning(
                    f"Not enough distinct models per family for grouped "
                    f"cross-validation (family '{limiting_family}' has only "
                    f"{n_groups}). Collect fingerprints for another model in that "
                    f"family, or pass groups=None to accept an optimistic estimate."
                )
            else:
                logger.warning("Not enough samples per class for cross-validation")
            return None

        if grouped:
            split_iter = self._grouped_splits(y, groups, actual_folds, random_state)
            logger.info(
                f"Grouped cross-validation: {actual_folds} folds "
                f"(capped by family '{limiting_family}' with {n_groups} distinct models)"
            )
        else:
            splitter = StratifiedKFold(
                n_splits=actual_folds, shuffle=True, random_state=random_state)
            split_iter = splitter.split(X, y)

        fold_accuracies = []
        argmax_fold_accuracies = []
        accepted = 0
        accepted_correct = 0
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create a temporary classifier with same settings
            temp = EnsembleClassifier(
                model_families=self.model_families,
                use_pca=self.use_pca,
                pca_components=self.pca_target_components,
                augment_data=self.augment_data,
                augment_noise_std=self.augment_noise_std,
                augment_samples=self.augment_samples,
                early_stop_variants=self.early_stop_variants,
                embedding_pca_dim=self.embedding_pca_dim,
                n_layers=self.n_layers,
                ood_confidence_threshold=self.ood_confidence_threshold,
                ood_disagreement_threshold=self.ood_disagreement_threshold,
                random_state=self.random_state,
            )
            if not temp.train(X_train, y_train):
                logger.error(f"Cross-validation training failed in fold {fold_idx + 1}")
                return None
            pipeline = IdentificationPipeline(temp)

            correct = 0
            argmax_correct = 0
            for i in range(len(X_val)):
                decision = pipeline.classify(X_val[i])
                family = decision.get("ensemble_result", {}).get("predicted_family")
                argmax_correct += self.model_families.get(family, -1) == int(y_val[i])
                pred_id = self.model_families.get(decision.get("family"), -1)
                all_y_true.append(int(y_val[i]))
                all_y_pred.append(pred_id)
                if pred_id >= 0:
                    accepted += 1
                    accepted_correct += pred_id == int(y_val[i])
                if pred_id == int(y_val[i]):
                    correct += 1

            acc = correct / len(X_val)
            fold_accuracies.append(acc)
            argmax_fold_accuracies.append(argmax_correct / len(X_val))
            logger.info(f"Fold {fold_idx + 1}/{actual_folds}: accuracy={acc:.3f}")

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            all_y_true, all_y_pred, labels=list(range(self.n_classes)), zero_division=0
        )
        # Keep rejected examples visible; otherwise the confusion matrix silently
        # loses rows while per-family support still includes them.
        cm_labels = list(range(self.n_classes)) + [-1]
        cm = confusion_matrix(all_y_true, all_y_pred, labels=cm_labels)

        per_family = {}
        for class_id in range(self.n_classes):
            if class_id in self.families_inv:
                name = self.families_inv[class_id]
                per_family[name] = {
                    "precision": float(precision[class_id]),
                    "recall": float(recall[class_id]),
                    "f1": float(f1[class_id]),
                    "support": int(support[class_id]),
                }

        mean_acc = float(np.mean(fold_accuracies))
        logger.info(f"Cross-validation: {actual_folds}-fold mean accuracy = {mean_acc:.3f}")

        return {
            "fold_accuracies": fold_accuracies,
            "mean_accuracy": mean_acc,
            "per_family": per_family,
            "confusion_matrix": cm,
            "confusion_labels": [self.families_inv.get(i, str(i)) for i in cm_labels[:-1]] + ["unknown"],
            "n_folds": actual_folds,
            "grouped": grouped,
            "limiting_family": limiting_family,
            "n_groups": n_groups,
            "argmax_fold_accuracies": argmax_fold_accuracies,
            "argmax_mean_accuracy": float(np.mean(argmax_fold_accuracies)),
            "accepted_coverage": accepted / len(all_y_true),
            "accepted_accuracy": accepted_correct / accepted if accepted else None,
            "rejected_count": len(all_y_true) - accepted,
            "evaluation_policy": "shared_identification",
        }


## This function can be changed if you want to build another classifier
def create_classifier(model_families=None, use_pca=False, embedding_pca_dim=64, **kwargs):

    return EnsembleClassifier(model_families=model_families, use_pca=use_pca,
                              embedding_pca_dim=embedding_pca_dim, **kwargs)


def get_available_classifiers():
    return ['ensemble']
