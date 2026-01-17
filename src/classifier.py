import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
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
                 augment_samples = 5):
        
        self.model_families = model_families

        self.families_inv = {v: k for k, v in self.model_families.items()}
        self.n_classes = len(self.model_families)
        
        # PCA settings
        self.use_pca = use_pca
        self.pca_target_components = pca_components
        self.pca_components = None
        self.pca = None
        
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
        logger.info(f"Initialized EnsembleClassifier {mode}, {self.n_classes} classes")

    def _augment_samples(self, X: np.ndarray, y: np.ndarray):
        """Augment training data by adding Gaussian noise."""
        if not self.augment_data or len(X) == 0:
            return X, y
            
        X_aug_list = [X]
        y_aug_list = [y]
        
        for _ in range(self.augment_samples):
            noise = np.random.normal(0, self.augment_noise_std, X.shape)
            X_noisy = X + noise * np.abs(X)
            X_aug_list.append(X_noisy)
            y_aug_list.append(y)
        
        X_aug = np.vstack(X_aug_list)
        y_aug = np.concatenate(y_aug_list)
        
        logger.info(f"Augmented {len(X)} -> {len(X_aug)} samples")
        return X_aug, y_aug

    def _preprocess(self, X: np.ndarray, fit = False):
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.input_dim = X.shape[1]
            
            if self.use_pca:
                max_components = min(X_scaled.shape[0], X_scaled.shape[1])
                self.pca_components = min(self.pca_target_components, max_components)
                
                if self.pca_components < self.pca_target_components:
                    logger.warning(f"Reducing PCA: {self.pca_target_components} -> {self.pca_components}")
                
                self.pca = PCA(n_components=self.pca_components)
                X_out = self.pca.fit_transform(X_scaled)
                variance = self.pca.explained_variance_ratio_.sum()
                logger.info(f"PCA: {X_out.shape[1]} components, {variance:.1%} variance")
            else:
                X_out = X_scaled
                logger.info(f"Using raw features: {X_out.shape[1]} dimensions")
        else:
            X_scaled = self.scaler.transform(X)
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
            self.mlp.fit(X_processed, y_train)

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
            return None, 0.0, {}

        try:
            if fingerprint.ndim == 1:
                fingerprint = fingerprint.reshape(1, -1)
            
            # Preprocess (same pipeline as training)
            fp_processed = self._preprocess(fingerprint, fit=False)
            logger.debug(f"Processed fingerprint shape: {fp_processed.shape}")
            # Get predictions from each classifier
            rf_pred = self.rf.predict_proba(fp_processed)[0]
            svm_pred = self.svm.predict_proba(fp_processed)[0]
            mlp_pred = self.mlp.predict_proba(fp_processed)[0]

            # MLP can be removed
            # logger.info(f"Result RF {rf_pred}, svm {svm_pred} and mlp {mlp_pred}")
            # Weighted ensemble
            logger.debug(f"RF prediction: {rf_pred}")
            logger.debug(f"SVM prediction: {svm_pred}")
            logger.debug(f"MLP prediction: {mlp_pred}")
            weights = [0.45, 0.45, 0.10]  # RF, SVM, MLP
            ensemble_pred = (weights[0] * rf_pred +
                           weights[1] * svm_pred +
                           weights[2] * mlp_pred)
            logger.debug(f"Ensemble prediction: {ensemble_pred}")
            top_idx = np.argmax(ensemble_pred)
            
            if top_idx not in self.families_inv:
                logger.error(f"Predicted class {top_idx} not in family mapping")
                return None, 0.0, {}
                
            top_family = self.families_inv[top_idx]
            top_confidence = ensemble_pred[top_idx]

            # Build probability dict
            probs = {}
            for i in range(len(ensemble_pred)):
                if i in self.families_inv:
                    probs[self.families_inv[i]] = float(ensemble_pred[i])

            return top_family, float(top_confidence), probs

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, 0.0, {}

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
            }
            joblib.dump(data, filepath)
            mode = "with PCA" if self.use_pca else "raw features"
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
            
            self.families_inv = {v: k for k, v in self.model_families.items()}
            self.n_classes = len(self.model_families)
            
            mode = "with PCA" if self.use_pca else "raw features"
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


## This function can be changed if you want to build another classifier
def create_classifier(model_families = None, use_pca = False, **kwargs):

    return EnsembleClassifier(model_families=model_families, use_pca=use_pca, **kwargs)


def get_available_classifiers():
    return ['ensemble']