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
                 ood_disagreement_threshold = 0.15):

        self.model_families = model_families

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

        # OOD detection settings
        self.ood_confidence_threshold = ood_confidence_threshold
        self.ood_disagreement_threshold = ood_disagreement_threshold

        # Augmentation settings
        self.augment_data = augment_data
        self.augment_noise_std = augment_noise_std
        self.augment_samples = augment_samples

        # Classifiers
        self.rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.svm = SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )

        self.scaler = StandardScaler()
        self.is_trained = False
        self.input_dim = None

        mode = "with PCA" if use_pca else "without PCA"
        logger.info(f"Initialized EnsembleClassifier {mode}, {self.n_classes} classes, "
                    f"embedding rebalancing {embedding_pca_dim}d")

    def _augment_samples(self, X: np.ndarray, y: np.ndarray):
        """Augment training data with additive Gaussian noise scaled to each
        feature's standard deviation, clipped to the observed per-feature range.

        Using std-scaled additive noise (not multiplicative) ensures that
        near-zero features (e.g. refusal_score for non-refusing models) are
        still perturbed — multiplicative noise would leave them unchanged.
        """
        if not self.augment_data or len(X) == 0:
            return X, y

        X_min  = X.min(axis=0)
        X_max  = X.max(axis=0)
        X_std  = X.std(axis=0)          # per-feature spread of real training data
        # Fall back to a small absolute value where std is zero (constant feature)
        X_std  = np.where(X_std > 0, X_std, 1e-6)

        X_aug_list = [X]
        y_aug_list = [y]

        for _ in range(self.augment_samples):
            noise    = np.random.normal(0, self.augment_noise_std, X.shape)
            X_noisy  = X + noise * X_std
            # Clip to observed range to prevent impossible values
            X_noisy  = np.clip(X_noisy, X_min, X_max)
            X_aug_list.append(X_noisy)
            y_aug_list.append(y)

        X_aug = np.vstack(X_aug_list)
        y_aug = np.concatenate(y_aug_list)

        logger.info(f"Augmented {len(X)} → {len(X_aug)} samples")
        return X_aug, y_aug

    def _rebalance_features(self, X: np.ndarray, fit: bool = False):
        """Compress embedding dimensions per layer block to rebalance feature groups.

        Input: (n_samples, n_layers * per_layer_dim) e.g. (N, 1206)
        Output: (n_samples, n_layers * (embedding_pca_dim + non_embed_dim)) e.g. (N, 246)
        """
        n_features = X.shape[1]
        assert n_features % self.n_layers == 0, \
            f"Feature dim {n_features} not divisible by n_layers {self.n_layers}"
        per_layer_dim = n_features // self.n_layers
        non_embed_dim = config.LINGUISTIC_DIM + config.BEHAVIORAL_DIM  # 18
        embed_dim = per_layer_dim - non_embed_dim

        if fit:
            self.per_layer_block_size = per_layer_dim
            self.per_layer_embed_dim = embed_dim
            actual_components = min(self.embedding_pca_dim, X.shape[0], embed_dim)
            if actual_components < self.embedding_pca_dim:
                logger.warning(f"Reducing embedding PCA: {self.embedding_pca_dim} -> {actual_components}")
            self.embedding_pcas = [
                PCA(n_components=actual_components) for _ in range(self.n_layers)
            ]

        if self.embedding_pcas is None:
            return X

        rebalanced_blocks = []
        for layer_idx in range(self.n_layers):
            start = layer_idx * per_layer_dim
            embeddings = X[:, start:start + embed_dim]
            non_embeddings = X[:, start + embed_dim:start + per_layer_dim]

            if fit:
                compressed = self.embedding_pcas[layer_idx].fit_transform(embeddings)
                variance = self.embedding_pcas[layer_idx].explained_variance_ratio_.sum()
                logger.info(f"Layer {layer_idx} embedding PCA: {compressed.shape[1]} components, "
                           f"{variance:.1%} variance retained")
            else:
                compressed = self.embedding_pcas[layer_idx].transform(embeddings)

            rebalanced_blocks.append(np.hstack([compressed, non_embeddings]))

        return np.hstack(rebalanced_blocks)

    def _preprocess(self, X: np.ndarray, fit = False):
        if fit:
            self.input_dim = X.shape[1]

            # Step 1: Rebalance features (compress embeddings per layer)
            X_rebalanced = self._rebalance_features(X, fit=True)
            logger.info(f"Rebalanced: {X.shape[1]} -> {X_rebalanced.shape[1]} dimensions")

            # Step 2: Scale
            X_scaled = self.scaler.fit_transform(X_rebalanced)

            # Step 3: Optional global PCA
            if self.use_pca:
                max_components = min(X_scaled.shape[0], X_scaled.shape[1])
                self.pca_components = min(self.pca_target_components, max_components)

                if self.pca_components < self.pca_target_components:
                    logger.warning(f"Reducing global PCA: {self.pca_target_components} -> {self.pca_components}")

                self.pca = PCA(n_components=self.pca_components)
                X_out = self.pca.fit_transform(X_scaled)
                variance = self.pca.explained_variance_ratio_.sum()
                logger.info(f"Global PCA: {X_out.shape[1]} components, {variance:.1%} variance")
            else:
                X_out = X_scaled
                logger.info(f"Using rebalanced features: {X_out.shape[1]} dimensions")
        else:
            X_rebalanced = self._rebalance_features(X, fit=False)
            X_scaled = self.scaler.transform(X_rebalanced)
            if self.use_pca and self.pca is not None:
                X_out = self.pca.transform(X_scaled)
            else:
                X_out = X_scaled

        return X_out

    def train(self, X: np.ndarray, y: np.ndarray):
        if X.shape[0] == 0:
            logger.error("Empty training data")
            return False

        if X.shape[0] < self.n_classes:
            logger.warning(f"Only {X.shape[0]} samples for {self.n_classes} classes")

        logger.info(f"Training on {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"PCA mode: {'enabled' if self.use_pca else 'disabled (raw features)'}")

        try:
            # Augment data
            X_train, y_train = self._augment_samples(X, y)
            
            # Preprocess (scale, optionally PCA)
            X_processed = self._preprocess(X_train, fit=True)
            
            logger.info(f"Training features shape: {X_processed.shape}")

            # Train classifiers
            logger.info("Training Random Forest...")
            self.rf.fit(X_processed, y_train)
            
            logger.info("Training SVM...")
            self.svm.fit(X_processed, y_train)
            
            logger.info("Training MLP...")
            # MLPClassifier has no class_weight param — pass balanced sample
            # weights so it gets the same imbalance handling as RF and SVM.
            classes, counts = np.unique(y_train, return_counts=True)
            class_weight = len(y_train) / (len(classes) * counts)
            sample_weight = np.array([class_weight[np.where(classes == c)[0][0]]
                                      for c in y_train])
            self.mlp.fit(X_processed, y_train, sample_weight=sample_weight)

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

            # Get predictions from each classifier
            rf_pred = self.rf.predict_proba(fp_processed)[0]
            svm_pred = self.svm.predict_proba(fp_processed)[0]
            mlp_pred = self.mlp.predict_proba(fp_processed)[0]

            logger.debug(f"RF prediction: {rf_pred}")
            logger.debug(f"SVM prediction: {svm_pred}")
            logger.debug(f"MLP prediction: {mlp_pred}")

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

            # Build probability dict
            probs = {}
            for i in range(len(ensemble_pred)):
                if i in self.families_inv:
                    probs[self.families_inv[i]] = float(ensemble_pred[i])

            # OOD detection: check confidence and classifier agreement
            top_per_classifier = [np.argmax(rf_pred), np.argmax(svm_pred), np.argmax(mlp_pred)]
            agreement_ratio = sum(1 for t in top_per_classifier if t == top_idx) / 3.0
            confidence_per_classifier = [rf_pred[top_idx], svm_pred[top_idx], mlp_pred[top_idx]]
            confidence_std = float(np.std(confidence_per_classifier))

            is_ood = (top_confidence < self.ood_confidence_threshold or
                      (agreement_ratio < 0.67 and confidence_std > self.ood_disagreement_threshold))

            ood_info = {
                'is_ood': is_ood,
                'agreement_ratio': agreement_ratio,
                'confidence_std': confidence_std,
                'classifier_top_classes': {
                    'rf': self.families_inv.get(top_per_classifier[0], '?'),
                    'svm': self.families_inv.get(top_per_classifier[1], '?'),
                    'mlp': self.families_inv.get(top_per_classifier[2], '?'),
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
        """Load trained classifier from file."""
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

        logger.info(f"Training from simulations: {family_counts}")
        return self.train(X, y)


    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5):
        """Run k-fold cross-validation and return per-family metrics.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Label vector (n_samples,).
            n_folds: Number of folds (default 5).

        Returns:
            Dict with keys:
            - fold_accuracies: list of per-fold accuracy
            - mean_accuracy: float
            - per_family: dict mapping family name -> {precision, recall, f1}
            - confusion_matrix: np.ndarray
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

        unique_classes = np.unique(y)
        actual_folds = min(n_folds, min(np.bincount(y.astype(int))))
        if actual_folds < 2:
            logger.warning("Not enough samples per class for cross-validation")
            return None

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

        fold_accuracies = []
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
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
            )
            temp.train(X_train, y_train)

            correct = 0
            for i in range(len(X_val)):
                family, _, _, _ = temp.predict_with_confidence(X_val[i])
                pred_id = self.model_families.get(family, -1)
                all_y_true.append(int(y_val[i]))
                all_y_pred.append(pred_id)
                if pred_id == int(y_val[i]):
                    correct += 1

            acc = correct / len(X_val)
            fold_accuracies.append(acc)
            logger.info(f"Fold {fold_idx + 1}/{actual_folds}: accuracy={acc:.3f}")

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            all_y_true, all_y_pred, labels=list(range(self.n_classes)), zero_division=0
        )
        cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(self.n_classes)))

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
            "n_folds": actual_folds,
        }


## This function can be changed if you want to build another classifier
def create_classifier(model_families=None, use_pca=False, embedding_pca_dim=64, **kwargs):

    return EnsembleClassifier(model_families=model_families, use_pca=use_pca,
                              embedding_pca_dim=embedding_pca_dim, **kwargs)


def get_available_classifiers():
    return ['ensemble']