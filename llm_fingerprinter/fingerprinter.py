import numpy as np
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMFingerprinter:
    def __init__(self, endpoint: str, ollama_client, prompt_suite,
                 feature_extractor, classifier):
        """
        Initialize fingerprinter.

        Args:
            endpoint: API endpoint URL
            ollama_client: Client instance (OllamaClient, OpenAIClient, etc.)
            prompt_suite: PromptSuite instance
            feature_extractor: FeatureExtractor instance
            classifier: EnsembleClassifier instance
        """
        self.endpoint = endpoint
        self.client = ollama_client
        self.suite = prompt_suite
        self.extractor = feature_extractor
        self.classifier = classifier

        logger.info(f"Initialized LLMFingerprinter for {endpoint}")
        logger.info(f"  - Feature extractor dim: {feature_extractor.get_feature_dim()}")
        logger.info(f"  - Prompt suite size: {len(prompt_suite)}")

    def fingerprint_model(self, model_name, repeats= 1,
                         progress_callback = None,
                         max_errors = 10):
        """
        Execute full fingerprinting pipeline for a single model.

        Returns RAW features (no PCA). The classifier handles any 
        dimensionality reduction during training/prediction.

        Args:
            model_name: Name of model on API
            repeats: Number of repeats per prompt (default: 1)
            progress_callback: Optional callback(current, total) for progress updates
            max_errors: Maximum consecutive errors before aborting

        Returns:
            Dict with keys:
            - model: str
            - timestamp: str
            - vector: np.ndarray (raw 402-dim features)
            - raw_features: Dict with mean features by type
            - metadata: Dict with execution details
            - responses_sample: List of sample prompt/response pairs
        """
        start_time = time.time()
        logger.info(f"Starting fingerprinting of {model_name}")

        # Get all prompts
        prompts = self.suite.get_prompts()
        total_queries = len(prompts) * repeats
        
        all_responses = []
        all_features = []
        query_count = 0
        error_count = 0
        consecutive_errors = 0

        # Execute prompts
        logger.info(f"Executing {len(prompts)} prompts × {repeats} repeats = {total_queries} queries")

        for i, prompt_dict in enumerate(prompts):
            prompt = prompt_dict['text']
            layer = prompt_dict.get('layer', 'unknown')
            logger.debug(f"Prompt {i+1}/{len(prompts)} (layer: {layer})")
            for rep in range(repeats):
                try:
                    logger.debug(f"model: {model_name}, prompt {i+1}, repeat {rep+1}")
                    response = self.client.generate(
                        model=model_name,
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=512
                    )
                    all_responses.append({
                        'prompt': prompt,
                        'response': response,
                        'layer': layer,
                        'category': prompt_dict.get('category', 'unknown'),
                    })
                    
                    # Debug logging (optional)
                    # debug_dir = f"./fingerprints/{model_name}"
                    # if not os.path.exists(debug_dir):
                    #    os.makedirs(debug_dir)
                    # with open(f"{debug_dir}/debug.txt", "a") as debug_file:
                    #    debug_file.write(f"PROMPT: {prompt}\n")
                    #    debug_file.write(f"RESPONSE: {response}\n\n")
                    
                    # Extract features
                    features = self.extractor.extract(prompt, response)
                    all_features.append(features)
                    query_count += 1
                    consecutive_errors = 0

                    # Progress update
                    if progress_callback:
                        progress_callback(query_count, total_queries)
                    elif (query_count % 10) == 0:
                        logger.info(f"Progress: {query_count}/{total_queries} queries "
                                   f"({query_count/total_queries*100:.1f}%)")

                except Exception as e:
                    error_count += 1
                    consecutive_errors += 1
                    logger.error(f"Error on prompt {i+1}, repeat {rep+1}: {e}")
                    
                    if consecutive_errors >= max_errors:
                        logger.error(f"Too many consecutive errors ({max_errors}), aborting")
                        break
                    continue
            
            if consecutive_errors >= max_errors:
                break

        # Aggregate features
        if len(all_features) == 0:
            logger.error("No features extracted!")
            return None

        # Group features by prompt layer for per-layer aggregation
        layer_order = ['stylistic', 'behavioral', 'discriminative']
        layer_features = {layer: [] for layer in layer_order}
        for feat, resp_info in zip(all_features, all_responses):
            layer = resp_info.get('layer', 'unknown')
            if layer in layer_features:
                layer_features[layer].append(feat)
            else:
                logger.warning(f"Unknown layer '{layer}', assigning to stylistic")
                layer_features['stylistic'].append(feat)

        # Average per layer, then concatenate
        feat_dim = self.extractor.get_feature_dim()
        layer_averages = []
        for layer_name in layer_order:
            feats = layer_features[layer_name]
            if feats:
                layer_avg = np.mean(feats, axis=0).astype(np.float32)
            else:
                logger.warning(f"No features for layer '{layer_name}', using zeros")
                layer_avg = np.zeros(feat_dim, dtype=np.float32)
            layer_averages.append(layer_avg)

        fingerprint_vector = np.concatenate(layer_averages)
        logger.info(f"Fingerprint vector: {fingerprint_vector.shape[0]} dimensions "
                    f"({len(layer_order)} layers x {feat_dim} features)")

        # Calculate feature indices for per-layer raw features breakdown
        embed_dim = self.extractor.embedding_dim
        ling_dim = self.extractor.LINGUISTIC_DIM
        behav_dim = self.extractor.BEHAVIORAL_DIM

        # Package results
        elapsed = time.time() - start_time

        # Build per-layer raw features
        raw_features = {}
        for layer_name in layer_order:
            feats = layer_features[layer_name]
            if feats:
                feats_array = np.array(feats, dtype=np.float32)
                raw_features[layer_name] = {
                    'embeddings': feats_array[:, :embed_dim].mean(axis=0).astype(np.float32),
                    'linguistic': feats_array[:, embed_dim:embed_dim+ling_dim].mean(axis=0).astype(np.float32),
                    'behavioral': feats_array[:, embed_dim+ling_dim:embed_dim+ling_dim+behav_dim].mean(axis=0).astype(np.float32),
                }
            else:
                raw_features[layer_name] = {
                    'embeddings': np.zeros(embed_dim, dtype=np.float32),
                    'linguistic': np.zeros(ling_dim, dtype=np.float32),
                    'behavioral': np.zeros(behav_dim, dtype=np.float32),
                }

        result = {
            'model': model_name,
            'timestamp': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            'vector': fingerprint_vector.astype(np.float32),
            'raw_features': raw_features,
            'metadata': {
                'endpoint': self.endpoint,
                'duration_seconds': round(elapsed, 2),
                'queries_executed': query_count,
                'queries_failed': error_count,
                'success_rate': round(query_count / total_queries, 3) if total_queries > 0 else 0,
                'feature_dim': fingerprint_vector.shape[0],
                'prompt_count': len(prompts),
                'repeats': repeats,
                'layers': {name: len(layer_features[name]) for name in layer_order},
            },
            'responses_sample': all_responses[:5],
        }

        logger.info(f"Fingerprinting complete in {elapsed:.1f}s "
                   f"({query_count}/{total_queries} queries, {fingerprint_vector.shape[0]} dims)")
        
        return result

    def run_simulations(self, model_name, num_simulations = 3,
                       repeats = 2, family = None):
        """
        Run multiple fingerprinting simulations for a model.

        Args:
            model_name: Name of model on API
            num_simulations: Number of independent fingerprinting runs
            repeats: Number of repeats per prompt in each simulation
            family: Optional family label (for logging)

        Returns:
            Dict mapping model name to list of fingerprint vectors
        """
        family_str = f" ({family})" if family else ""
        logger.info(f"Starting {num_simulations} simulations for {model_name}{family_str}")
        
        vectors = []
        metadata_list = []

        for sim_idx in range(num_simulations):
            logger.info(f"  Simulation {sim_idx + 1}/{num_simulations}")
            
            fp = self.fingerprint_model(model_name, repeats=repeats)

            if fp is None:
                logger.warning(f"  Simulation {sim_idx + 1} failed, skipping")
                continue

            vectors.append(fp['vector'])
            metadata_list.append(fp['metadata'])
            
            logger.info(f"  Simulation {sim_idx + 1} complete: "
                       f"{fp['metadata']['queries_executed']} queries, "
                       f"{fp['metadata']['duration_seconds']:.1f}s")

        logger.info(f"Completed {len(vectors)}/{num_simulations} simulations for {model_name}")
        
        return {model_name: vectors}

    def identify(self, model_name: str, repeats= 1):
        """
        Identify model family using trained classifier.

        Args:
            model_name: Model to identify
            repeats: Number of prompt repeats (default: 1)

        Returns:
            Dict with classification results:
            - model: str
            - family: str (predicted family)
            - confidence: float (0-1)
            - all_probabilities: Dict[str, float]
            - fingerprint: Dict (full fingerprint data)
            - error: str (if failed)
        """
        # Check if classifier is trained
        if not self.classifier.is_trained:
            return {
                'model': model_name,
                'error': 'Classifier not trained. Run training first.'
            }

        # Generate fingerprint (raw features, no PCA)
        logger.info(f"Fingerprinting {model_name} for identification...")
        fp = self.fingerprint_model(model_name, repeats=repeats)

        if fp is None:
            return {
                'model': model_name,
                'error': 'Fingerprinting failed'
            }

        # Classify using raw features - classifier handles rebalancing/PCA internally
        family, confidence, all_probs, ood_info = self.classifier.predict_with_confidence(fp['vector'])

        if family is None:
            return {
                'model': model_name,
                'error': 'Classification failed',
                'fingerprint': fp
            }

        is_ood = ood_info.get('is_ood', False)
        result = {
            'model': model_name,
            'family': 'unknown' if is_ood else family,
            'predicted_family': family,
            'confidence': round(confidence, 4),
            'all_probabilities': {k: round(v, 4) for k, v in all_probs.items()},
            'ood_detected': is_ood,
            'ood_details': ood_info,
            'fingerprint': fp,
        }

        if is_ood:
            logger.warning(f"OOD detected for {model_name}: best guess {family} "
                          f"({confidence*100:.1f}% confidence)")
        else:
            logger.info(f"Identified as {family} ({confidence*100:.1f}% confidence)")
        return result
