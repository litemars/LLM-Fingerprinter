import numpy as np
import logging
import time
from datetime import datetime

from llm_fingerprinter import config

logger = logging.getLogger(__name__)


class LLMFingerprinter:
    def __init__(self, endpoint: str, ollama_client, prompt_suite,
                 feature_extractor, classifier):
        """
        Initialize fingerprinter.

        Args:
            endpoint:         API endpoint URL
            ollama_client:    Client instance (OllamaClient, OpenAIClient, etc.)
            prompt_suite:     PromptSuite instance
            feature_extractor: FeatureExtractor instance
            classifier:       EnsembleClassifier instance
        """
        self.endpoint = endpoint
        self.client = ollama_client
        self.suite = prompt_suite
        self.extractor = feature_extractor
        self.classifier = classifier

        logger.info(f"Initialized LLMFingerprinter for {endpoint}")
        logger.info(f"  - Feature extractor dim: {feature_extractor.get_feature_dim()}")
        logger.info(f"  - Prompt suite size: {len(prompt_suite)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_partial_fingerprint(self,
                                   layer_features: dict,
                                   completed_layers: list,
                                   layer_order: list) -> np.ndarray:
        """Build a full-dimensional fingerprint for early-stopping confidence checks.

        Completed layers use their actual averaged features.
        Missing layers are padded with the mean of completed layers — the best
        available approximation that keeps the vector classifier-compatible.

        Args:
            layer_features:   dict mapping layer name -> list of per-prompt feature vecs
            completed_layers: layers that have been fully executed so far
            layer_order:      canonical concatenation order (from config.LAYER_ORDER)

        Returns:
            np.ndarray of shape (n_layers * feat_dim,)
        """
        feat_dim = self.extractor.get_feature_dim()

        completed_avgs = {
            layer: np.mean(layer_features[layer], axis=0).astype(np.float32)
            for layer in completed_layers
            if layer_features.get(layer)
        }

        if not completed_avgs:
            return np.zeros(feat_dim * len(layer_order), dtype=np.float32)

        fallback = np.mean(list(completed_avgs.values()), axis=0).astype(np.float32)

        return np.concatenate([
            completed_avgs.get(layer, fallback)
            for layer in layer_order
        ])

    def fingerprint_model(self, model_name, repeats=1,
                          progress_callback=None,
                          max_errors=10,
                          early_stop_confidence=None,
                          temperature=None):
        """Execute full fingerprinting pipeline for a single model.

        Prompts run layer-by-layer in config.LAYER_ORDER order:
            discriminative → behavioral → stylistic

        After each layer completes, if `early_stop_confidence` is set and the
        classifier is trained, confidence is checked and execution may stop early,
        saving API calls for clear-cut identifications.

        NOTE: `early_stop_confidence` is intentionally NOT passed by
        `run_simulations` — training always uses all layers.

        Args:
            model_name:             Name of model on the API
            repeats:                Prompt repeats per query (default: 1)
            progress_callback:      Optional callback(current, total)
            max_errors:             Max consecutive errors before aborting
            early_stop_confidence:  Float in [0,1]. Stop after a layer if
                                    classifier confidence >= this value.
                                    None = disabled (always run all layers).
            temperature:            Sampling temperature (default: config.TEMPERATURE).

        Returns:
            Dict with keys:
              model, timestamp, vector, raw_features, metadata, responses_sample
            metadata includes: early_stopped (bool), layers_completed (list),
                                queries_total (int)
        """
        temperature = temperature if temperature is not None else config.TEMPERATURE
        start_time = time.time()
        logger.info(f"Starting fingerprinting of {model_name} (temp={temperature:.2f})")

        layer_order = config.LAYER_ORDER  # ['discriminative', 'behavioral', 'stylistic']

        prompts_by_layer = {
            layer: self.suite.get_prompts(layer=layer)
            for layer in layer_order
        }
        total_queries = sum(len(prompts_by_layer[l]) for l in layer_order) * repeats

        all_responses = []
        layer_features = {layer: [] for layer in layer_order}
        completed_layers = []
        query_count = 0
        error_count = 0
        consecutive_errors = 0
        early_stopped = False

        mode_str = f"early-stop @ {early_stop_confidence}" if early_stop_confidence else "full suite"
        logger.info(f"Executing {total_queries} queries across {len(layer_order)} layers "
                    f"[{mode_str}]")

        for layer_name in layer_order:
            layer_prompts = prompts_by_layer[layer_name]
            logger.debug(f"Layer '{layer_name}': {len(layer_prompts)} prompts × {repeats} repeats")

            # ── Phase 1: collect API responses (sequential) ───────────────────
            layer_pairs = []   # (prompt_text, response, prompt_dict)
            for prompt_dict in layer_prompts:
                prompt = prompt_dict['text']
                for rep in range(repeats):
                    try:
                        response = self.client.generate(
                            model=model_name,
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=512
                        )
                        layer_pairs.append((prompt, response, prompt_dict))
                        query_count += 1
                        consecutive_errors = 0

                        if progress_callback:
                            progress_callback(query_count, total_queries)
                        elif query_count % 10 == 0:
                            logger.info(f"Progress: {query_count}/{total_queries} queries "
                                        f"({query_count / total_queries * 100:.1f}%)")

                    except Exception as e:
                        error_count += 1
                        consecutive_errors += 1
                        logger.error(
                            f"Error in layer '{layer_name}', repeat {rep + 1}: {e}"
                        )
                        if consecutive_errors >= max_errors:
                            logger.error(
                                f"Too many consecutive errors ({max_errors}), aborting"
                            )
                            break
                        continue

                if consecutive_errors >= max_errors:
                    break

            # ── Phase 2: batch-extract features (single embedding forward pass)
            if layer_pairs:
                features_batch = self.extractor.extract_batch(
                    [(p, r) for p, r, _ in layer_pairs]
                )
                for (prompt, response, pd), features in zip(layer_pairs, features_batch):
                    all_responses.append({
                        'prompt': prompt,
                        'response': response,
                        'layer': layer_name,
                        'category': pd.get('category', 'unknown'),
                    })
                    layer_features[layer_name].append(features)

            completed_layers.append(layer_name)
            logger.info(f"Layer '{layer_name}' done "
                        f"({len(layer_prompts) * repeats} queries, "
                        f"total so far: {query_count})")

            if (early_stop_confidence is not None
                    and self.classifier is not None
                    and self.classifier.is_trained
                    and layer_features[layer_name]):

                partial_vec = self._build_partial_fingerprint(
                    layer_features, completed_layers, layer_order
                )
                _, confidence, _, _ = self.classifier.predict_with_confidence(partial_vec)

                logger.info(
                    f"Early-stop check after '{layer_name}': "
                    f"confidence={confidence:.3f} (threshold={early_stop_confidence})"
                )

                if confidence >= early_stop_confidence:
                    saved = total_queries - query_count
                    logger.info(
                        f"Early stop triggered — used {query_count}/{total_queries} queries "
                        f"({query_count / total_queries * 100:.0f}%), saved {saved} queries"
                    )
                    early_stopped = True
                    break

            if consecutive_errors >= max_errors:
                break

        # ── Build final fingerprint ────────────────────────────────────────────
        if query_count == 0:
            logger.error("No features extracted!")
            return None

        feat_dim = self.extractor.get_feature_dim()
        embed_dim = self.extractor.embedding_dim
        ling_dim = self.extractor.LINGUISTIC_DIM
        behav_dim = self.extractor.BEHAVIORAL_DIM

        completed_avgs = {
            layer: np.mean(layer_features[layer], axis=0).astype(np.float32)
            for layer in completed_layers
            if layer_features.get(layer)
        }
        fallback = (
            np.mean(list(completed_avgs.values()), axis=0).astype(np.float32)
            if completed_avgs else np.zeros(feat_dim, dtype=np.float32)
        )

        layer_averages = []
        raw_features = {}
        for layer_name in layer_order:
            if layer_features[layer_name]:
                layer_avg = completed_avgs[layer_name]
            else:
                layer_avg = fallback
                logger.warning(f"Layer '{layer_name}' not executed — using fallback average")

            layer_averages.append(layer_avg)
            raw_features[layer_name] = {
                'embeddings': layer_avg[:embed_dim],
                'linguistic': layer_avg[embed_dim:embed_dim + ling_dim],
                'behavioral': layer_avg[embed_dim + ling_dim:embed_dim + ling_dim + behav_dim],
            }

        fingerprint_vector = np.concatenate(layer_averages)
        elapsed = time.time() - start_time

        logger.info(
            f"Fingerprinting complete in {elapsed:.1f}s — "
            f"{query_count}/{total_queries} queries used, "
            f"{'early stopped after: ' + str(completed_layers) if early_stopped else 'all layers'}, "
            f"{fingerprint_vector.shape[0]} dims"
        )

        return {
            'model': model_name,
            'timestamp': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            'vector': fingerprint_vector.astype(np.float32),
            'raw_features': raw_features,
            'metadata': {
                'endpoint': self.endpoint,
                'duration_seconds': round(elapsed, 2),
                'queries_executed': query_count,
                'queries_total': total_queries,
                'queries_failed': error_count,
                'success_rate': round(query_count / total_queries, 3) if total_queries > 0 else 0,
                'feature_dim': fingerprint_vector.shape[0],
                'prompt_count': sum(len(prompts_by_layer[l]) for l in layer_order),
                'temperature': round(temperature, 2),
                'repeats': repeats,
                'layers_completed': completed_layers,
                'early_stopped': early_stopped,
                'layers': {name: len(layer_features[name]) for name in layer_order},
            },
            'responses_sample': all_responses[:5],
        }

    def run_simulations(self, model_name, num_simulations=3,
                        repeats=2, family=None):
        """Run multiple independent fingerprinting simulations for training.

        Always runs all layers — early stopping is intentionally disabled here
        to ensure complete, consistent training fingerprints.

        Args:
            model_name:        Name of model on API
            num_simulations:   Number of independent runs
            repeats:           Prompt repeats per simulation
            family:            Optional family label (for logging)

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
            # early_stop_confidence intentionally omitted — always full suite

            if fp is None:
                logger.warning(f"  Simulation {sim_idx + 1} failed, skipping")
                continue

            vectors.append(fp['vector'])
            metadata_list.append(fp['metadata'])

            logger.info(
                f"  Simulation {sim_idx + 1} complete: "
                f"{fp['metadata']['queries_executed']} queries, "
                f"{fp['metadata']['duration_seconds']:.1f}s"
            )

        logger.info(f"Completed {len(vectors)}/{num_simulations} simulations for {model_name}")
        return {model_name: vectors}

    def identify(self, model_name: str, repeats=1, early_stop_confidence=None):
        """Identify model family using the trained classifier.

        Args:
            model_name:             Model to identify
            repeats:                Prompt repeats (default: 1)
            early_stop_confidence:  Stop after a layer if confidence >= this value.
                                    Recommended range: 0.88–0.95.
                                    None = disabled (run all layers).

        Returns:
            Dict with classification results:
              model, family, predicted_family, confidence, all_probabilities,
              ood_detected, ood_details, early_stopped, layers_completed,
              queries_executed, queries_total, fingerprint
        """
        if not self.classifier.is_trained:
            return {
                'model': model_name,
                'error': 'Classifier not trained. Run training first.'
            }

        logger.info(
            f"Fingerprinting {model_name} for identification "
            f"(early_stop={'off' if early_stop_confidence is None else early_stop_confidence})"
        )

        fp = self.fingerprint_model(
            model_name,
            repeats=repeats,
            early_stop_confidence=early_stop_confidence
        )

        if fp is None:
            return {'model': model_name, 'error': 'Fingerprinting failed'}

        family, confidence, all_probs, ood_info = self.classifier.predict_with_confidence(
            fp['vector']
        )

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
            'early_stopped': fp['metadata'].get('early_stopped', False),
            'layers_completed': fp['metadata'].get('layers_completed', []),
            'queries_executed': fp['metadata'].get('queries_executed', 0),
            'queries_total': fp['metadata'].get('queries_total', 0),
            'fingerprint': fp,
        }

        if is_ood:
            logger.warning(
                f"OOD detected for {model_name}: best guess {family} "
                f"({confidence * 100:.1f}% confidence)"
            )
        else:
            logger.info(f"Identified as {family} ({confidence * 100:.1f}% confidence)")

        return result
