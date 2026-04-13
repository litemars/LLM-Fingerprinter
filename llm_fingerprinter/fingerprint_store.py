import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


class FingerprintStore:
    def __init__(self, basedir = "./fingerprints"):
        self.basedir = Path(basedir)
        self.basedir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized FingerprintStore at {self.basedir}")

    def _serialize_value(self, value):
        """Recursively serialize values for JSON compatibility."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        return value

    def save_fingerprint(self, fingerprint, model_name, family = None):

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ")

        # JSON to save
        data = {
            "model": model_name,
            "family": family,
            "timestamp": timestamp,
            "vector": self._serialize_value(fingerprint.get("vector", [])),
            "raw_features": self._serialize_value(fingerprint.get("raw_features", {})),
            "metadata": self._serialize_value(fingerprint.get("metadata", {})),
        }

        # Create safe filename (remove special characters)
        safe_model_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
        filename = f"{safe_model_name}_{timestamp}.json"
        filepath = self.basedir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved fingerprint to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save fingerprint: {e}")
            raise IOError(f"Failed to save fingerprint to {filepath}: {e}")

    def _load_file(self, filepath):
        """Load fingerprint from a file path."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert vector back to numpy array
            if 'vector' in data and data['vector']:
                data['vector'] = np.array(data['vector'], dtype=np.float32)
            
            # Convert raw_features arrays back to numpy
            if 'raw_features' in data:
                for key, value in data['raw_features'].items():
                    if isinstance(value, list):
                        data['raw_features'][key] = np.array(value, dtype=np.float32)
            
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load fingerprint from {filepath}: {e}")
            return None

    def load_fingerprint(self, filepath: str):
        """
        Load fingerprint from file.
        
        Args:
            filepath: Path to fingerprint file
            
        Returns:
            Fingerprint dict or None if not found/invalid
        """
        path = Path(filepath)

        if not path.exists():
            logger.warning(f"Fingerprint not found at {filepath}")
            return None

        return self._load_file(path)

    def list_baselines(self):
        """
        List all baseline fingerprints.
        
        Returns:
            Dict mapping family names to file paths
        """
        baselines = {}
        for f in self.basedir.glob("*_baseline.json"):
            family = f.stem.replace("_baseline", "")
            baselines[family] = f
        return baselines

    def list_fingerprints(self, model_name = None, family = None):
        """
        List all saved fingerprints, optionally filtered.
        
        Args:
            model_name: Filter by model name
            family: Filter by family (requires loading each file)
            
        Returns:
            List of file paths, sorted by most recent first
        """
        files = list(self.basedir.glob("*.json"))
        
        # Exclude baseline and model files
        files = [f for f in files if "_baseline" not in f.name and "_model" not in f.name]

        if model_name:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
            files = [f for f in files if safe_name in f.name or model_name in f.name]
        
        if family:
            # Need to load files to check family
            filtered = []
            for f in files:
                data = self.load_fingerprint(str(f))
                if data and data.get("family") == family:
                    filtered.append(f)
            files = filtered

        return sorted(files, reverse=True)

    # not called now
    def delete_fingerprint(self, filepath: str):
        path = Path(filepath)
        
        if not path.exists():
            logger.warning(f"File not found: {filepath}")
            return False
        
        try:
            path.unlink()
            logger.info(f"Deleted fingerprint: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {filepath}: {e}")
            return False

    def count_by_family(self):
        """Count fingerprints per family."""
        counts = {}
        
        for filepath in self.list_fingerprints():
            data = self.load_fingerprint(str(filepath))
            if data and data.get("family"):
                family = data["family"]
                counts[family] = counts.get(family, 0) + 1
        
        return counts

    def _get_full_vector(self, data):

        vector = data.get("vector")
        raw_features = data.get("raw_features", {})

        # Use stored vector directly if available
        if vector is not None and len(vector) >= 400:
            return vector

        # Try per-layer reconstruction (new format)
        layer_order = ['stylistic', 'behavioral', 'discriminative']
        if any(layer in raw_features for layer in layer_order):
            layer_vectors = []
            for layer_name in layer_order:
                layer_data = raw_features.get(layer_name, {})
                embeddings = layer_data.get("embeddings")
                linguistic = layer_data.get("linguistic")
                behavioral = layer_data.get("behavioral")

                if embeddings is None:
                    logger.warning(f"No embeddings for layer '{layer_name}'")
                    return None

                parts = []
                for arr in [embeddings, linguistic, behavioral]:
                    if arr is not None:
                        if not isinstance(arr, np.ndarray):
                            arr = np.array(arr, dtype=np.float32)
                        parts.append(arr)

                layer_vectors.append(np.concatenate(parts))

            full_vector = np.concatenate(layer_vectors)
            logger.debug(f"Reconstructed per-layer vector: {len(full_vector)} dims")
            return full_vector

        # Fallback: old flat format
        embeddings = raw_features.get("embeddings")
        linguistic = raw_features.get("linguistic")
        behavioral = raw_features.get("behavioral")

        if embeddings is None:
            logger.warning("No embeddings found in raw_features")
            return None

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        if linguistic is None or behavioral is None:
            logger.debug(f"Only embeddings available: {len(embeddings)} dims")
            return embeddings

        if not isinstance(linguistic, np.ndarray):
            linguistic = np.array(linguistic, dtype=np.float32)
        if not isinstance(behavioral, np.ndarray):
            behavioral = np.array(behavioral, dtype=np.float32)

        full_vector = np.concatenate([embeddings, linguistic, behavioral])
        logger.debug(f"Reconstructed full vector: {len(full_vector)} dims "
                    f"({len(embeddings)} + {len(linguistic)} + {len(behavioral)})")

        return full_vector

    def export_for_training(self):
        training_data = {}
        
        for filepath in self.list_fingerprints():
            data = self.load_fingerprint(str(filepath))
            if data and data.get("family"):
                family = data["family"]
                
                # Try to get or reconstruct full 402-dim vector
                vector = self._get_full_vector(data)
                
                if vector is None:
                    logger.warning(f"Skipping {filepath}: could not get full vector")
                    continue
                
                if family not in training_data:
                    training_data[family] = []
                training_data[family].append(vector)
                
                logger.debug(f"Loaded {family} fingerprint: {len(vector)} dims")
        
        for family, vectors in training_data.items():
            if vectors:
                logger.info(f"Exported {family}: {len(vectors)} samples, {len(vectors[0])} dims each")
        
        return training_data
