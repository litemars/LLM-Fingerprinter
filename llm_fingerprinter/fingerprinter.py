import numpy as np
import logging
import time
from datetime import datetime

from llm_fingerprinter import config
from llm_fingerprinter.identification import IdentificationPipeline
from llm_fingerprinter.feature_validation import (
    feature_schema, validate_response_features, validate_fingerprint_vector,
    current_prompt_suite_hash, validate_inference_metadata, FeatureValidationError,
)

logger = logging.getLogger(__name__)


class LLMFingerprinter:
    def __init__(self, endpoint: str, ollama_client, prompt_suite,
                 feature_extractor, classifier, family_templates=None, model_templates=None):
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
        self.family_templates = family_templates
        self.model_templates = model_templates

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
        """Full-dimensional fingerprint for early-stop confidence checks: completed
        layers use their averaged features, missing layers are padded with the mean
        of completed ones."""
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
                          temperature=None,
                          min_layer_coverage=None):
        """Execute full fingerprinting pipeline for a single model.

        Prompts run layer-by-layer in config.LAYER_ORDER order:
            discriminative → behavioral → stylistic

        After each layer completes, if `early_stop_confidence` is set and the
        classifier is trained, confidence is checked and execution may stop early,
        saving API calls for clear-cut identifications.

        Collection integrity rules:
          * A response that is empty or whitespace-only is a FAILED observation,
            not a successful one. Backends return "" for API error payloads,
            content filters and truncated reasoning output; counting those as
            successes silently drags the fingerprint toward the zero vector.
          * A layer that was attempted but yielded less than `min_layer_coverage`
            of its prompts causes the whole fingerprint to be rejected (None).
            Only layers deliberately SKIPPED by early stopping are padded — a
            layer that was asked and failed must never be reconstructed from
            other layers, because the result is indistinguishable from a
            complete fingerprint and is scored just as confidently.
          * Tripping `max_errors` consecutive failures rejects the fingerprint.

        NOTE: `early_stop_confidence` is intentionally NOT passed by
        `run_simulations` — training always uses all layers.

        Args:
            model_name:             Name of model on the API
            repeats:                Prompt repeats per query (default: 1)
            progress_callback:      Optional callback(current, total)
            max_errors:             Max consecutive errors before rejecting
            early_stop_confidence:  Float in [0,1]. Stop after a layer if
                                    classifier confidence >= this value.
                                    None = disabled (always run all layers).
            temperature:            Sampling temperature (default: config.TEMPERATURE).
            min_layer_coverage:     Minimum fraction of a layer's prompts that must
                                    return usable text (default:
                                    config.MIN_LAYER_COVERAGE).

        Returns:
            Dict with keys:
              model, timestamp, vector, raw_features, metadata, responses_sample
            metadata includes: early_stopped (bool), layers_completed (list),
                                layers_attempted, layers_skipped, layer_coverage,
                                queries_total (int)
            None if collection did not meet the quality gates.
        """
        temperature = temperature if temperature is not None else config.TEMPERATURE
        if min_layer_coverage is None:
            min_layer_coverage = config.MIN_LAYER_COVERAGE
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if not isinstance(max_errors, int) or max_errors < 1:
            raise ValueError("max_errors must be a positive integer")
        if not 0 < min_layer_coverage <= 1:
            raise ValueError("min_layer_coverage must be in (0, 1]")
        if early_stop_confidence is not None and not 0 < early_stop_confidence <= 1:
            raise ValueError("early_stop_confidence must be in (0, 1]")
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
        attempted_layers = []
        layer_coverage = {}
        query_count = 0
        attempted_count = 0
        error_count = 0
        empty_count = 0
        consecutive_errors = 0
        early_stopped = False

        mode_str = f"early-stop @ {early_stop_confidence}" if early_stop_confidence else "full suite"
        logger.info(f"Executing {total_queries} queries across {len(layer_order)} layers "
                    f"[{mode_str}, min layer coverage {min_layer_coverage:.0%}]")

        for layer_name in layer_order:
            layer_prompts = prompts_by_layer[layer_name]
            layer_expected = len(layer_prompts) * repeats
            attempted_layers.append(layer_name)
            logger.debug(f"Layer '{layer_name}': {len(layer_prompts)} prompts × {repeats} repeats")

            # ── Phase 1: collect API responses (sequential) ───────────────────
            layer_pairs = []   # (prompt_text, response, prompt_dict)
            aborted = False
            for prompt_dict in layer_prompts:
                prompt = prompt_dict['text']
                for rep in range(repeats):
                    attempted_count += 1
                    try:
                        response = self.client.generate(
                            model=model_name,
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=config.MAX_TOKENS
                        )
                    except Exception as e:
                        error_count += 1
                        consecutive_errors += 1
                        logger.error(
                            f"Error in layer '{layer_name}', repeat {rep + 1}: {e}"
                        )
                        if consecutive_errors >= max_errors:
                            aborted = True
                            break
                        continue

                    # An empty body is a failed observation, not a blank answer.
                    if not isinstance(response, str) or not response.strip():
                        error_count += 1
                        empty_count += 1
                        consecutive_errors += 1
                        logger.error(
                            f"Empty response in layer '{layer_name}', repeat {rep + 1} "
                            f"(prompt: {prompt[:60]!r}) — counted as a failed query"
                        )
                        if consecutive_errors >= max_errors:
                            aborted = True
                            break
                        continue

                    layer_pairs.append((prompt, response, prompt_dict))
                    query_count += 1
                    consecutive_errors = 0

                    if progress_callback:
                        progress_callback(query_count, total_queries)
                    elif query_count % 10 == 0:
                        logger.info(f"Progress: {query_count}/{total_queries} queries "
                                    f"({query_count / total_queries * 100:.1f}%)")

                if aborted:
                    break

            if aborted:
                logger.error(
                    f"Aborting: {max_errors} consecutive failed queries in layer "
                    f"'{layer_name}'. The backend is not usable — no fingerprint produced."
                )
                return None

            # ── Phase 2: batch-extract features (single embedding forward pass)
            if layer_pairs:
                try:
                    features_batch = self.extractor.extract_batch(
                        [(p, r) for p, r, _ in layer_pairs])
                    if len(features_batch) != len(layer_pairs):
                        raise ValueError("Feature extractor returned the wrong number of rows")
                    features_batch = [validate_response_features(
                        features, embedding_dim=self.extractor.embedding_dim,
                        linguistic_dim=self.extractor.LINGUISTIC_DIM,
                        behavioral_dim=self.extractor.BEHAVIORAL_DIM)
                        for features in features_batch]
                except Exception as error:
                    logger.error("Feature extraction failed in layer '%s': %s. "
                                 "No fingerprint produced.", layer_name, error)
                    return None
                for (prompt, response, pd), features in zip(layer_pairs, features_batch):
                    all_responses.append({
                        'prompt': prompt,
                        'response': response,
                        'layer': layer_name,
                        'category': pd.get('category', 'unknown'),
                    })
                    layer_features[layer_name].append(features)
                completed_layers.append(layer_name)

            coverage = len(layer_pairs) / layer_expected if layer_expected else 0.0
            layer_coverage[layer_name] = round(coverage, 3)

            if coverage < min_layer_coverage:
                logger.error(
                    f"Layer '{layer_name}' collected {len(layer_pairs)}/{layer_expected} "
                    f"responses ({coverage:.0%}), below the {min_layer_coverage:.0%} "
                    f"minimum. Rejecting the fingerprint rather than padding a failed "
                    f"layer with other layers' data."
                )
                return None

            logger.info(f"Layer '{layer_name}' done "
                        f"({len(layer_pairs)}/{layer_expected} responses, "
                        f"total so far: {query_count})")

            if (early_stop_confidence is not None
                    and self.classifier is not None
                    and self.classifier.is_trained
                    and layer_features[layer_name]
                    and layer_name != layer_order[-1]):

                partial_vec = self._build_partial_fingerprint(
                    layer_features, completed_layers, layer_order
                )
                _, confidence, _, ood = self.classifier.predict_with_confidence(partial_vec)

                logger.info(
                    f"Early-stop check after '{layer_name}': "
                    f"confidence={confidence:.3f} (threshold={early_stop_confidence})"
                )

                if confidence >= early_stop_confidence and not ood.get('is_ood', False):
                    saved = total_queries - query_count
                    logger.info(
                        f"Early stop triggered — used {query_count}/{total_queries} queries "
                        f"({query_count / total_queries * 100:.0f}%), saved {saved} queries"
                    )
                    early_stopped = True
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
        # Padding value for layers early stopping chose to SKIP. Every attempted
        # layer is guaranteed present here — insufficient coverage returned None
        # above — so this can only ever fill in deliberately skipped layers.
        fallback = (
            np.mean(list(completed_avgs.values()), axis=0).astype(np.float32)
            if completed_avgs else np.zeros(feat_dim, dtype=np.float32)
        )
        skipped_layers = [l for l in layer_order if l not in attempted_layers]

        layer_averages = []
        raw_features = {}
        for layer_name in layer_order:
            if layer_features[layer_name]:
                layer_avg = completed_avgs[layer_name]
            else:
                layer_avg = fallback
                logger.info(f"Layer '{layer_name}' skipped by early stopping — "
                            f"padded with the mean of completed layers")

            layer_averages.append(layer_avg)
            raw_features[layer_name] = {
                'embeddings': layer_avg[:embed_dim],
                'linguistic': layer_avg[embed_dim:embed_dim + ling_dim],
                'behavioral': layer_avg[embed_dim + ling_dim:embed_dim + ling_dim + behav_dim],
            }

        try:
            fingerprint_vector = validate_fingerprint_vector(
                np.concatenate(layer_averages), embedding_dim=embed_dim,
                linguistic_dim=ling_dim, behavioral_dim=behav_dim,
                layer_order=layer_order)
        except ValueError as error:
            logger.error("Invalid aggregate fingerprint: %s", error)
            return None
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
                'queries_attempted': attempted_count,
                'queries_total': total_queries,
                'queries_failed': error_count,
                'queries_empty': empty_count,
                # Of the queries actually issued, how many produced usable text.
                'success_rate': (round(query_count / attempted_count, 3)
                                 if attempted_count > 0 else 0),
                # Of the full 31-prompt suite, how much was covered. Differs from
                # success_rate when early stopping skipped layers.
                'suite_coverage': (round(query_count / total_queries, 3)
                                   if total_queries > 0 else 0),
                'feature_dim': fingerprint_vector.shape[0],
                'feature_schema': feature_schema(
                    embedding_model=getattr(self.extractor, 'model_name', config.EMBEDDING_MODEL),
                    embedding_dim=embed_dim, linguistic_dim=ling_dim,
                    behavioral_dim=behav_dim, layer_order=layer_order),
                'prompt_suite_hash': current_prompt_suite_hash(self.suite.get_prompts()),
                'max_tokens': config.MAX_TOKENS,
                'prompt_count': sum(len(prompts_by_layer[l]) for l in layer_order),
                'temperature': round(temperature, 2),
                'repeats': repeats,
                'layers_completed': completed_layers,
                'layers_attempted': attempted_layers,
                'layers_skipped': skipped_layers,
                'layer_coverage': layer_coverage,
                'min_layer_coverage': min_layer_coverage,
                'early_stopped': early_stopped,
                # True when part of the vector is padding rather than observation.
                'incomplete': bool(skipped_layers),
                'layers': {name: len(layer_features[name]) for name in layer_order},
            },
            'responses': all_responses,
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

            # early_stop_confidence intentionally omitted — always full suite.
            # Training data is held to TRAINING_MIN_LAYER_COVERAGE: a degraded
            # fingerprint saved under a family label poisons every later run.
            fp = self.fingerprint_model(
                model_name, repeats=repeats,
                min_layer_coverage=config.TRAINING_MIN_LAYER_COVERAGE)

            if fp is None:
                logger.warning(
                    f"  Simulation {sim_idx + 1} did not meet the collection "
                    f"quality gate, skipping")
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
            return {
                'model': model_name,
                'error': (
                    'Fingerprinting failed — too few usable responses to build a '
                    'reliable fingerprint. Padding the missing layers would produce '
                    'a confident but unfounded answer, so no result is reported. '
                    'See the log for which layer fell short.'
                ),
            }

        result = self.classify_fingerprint(fp)
        result['model'] = model_name
        logger.info("Identification: %s (%s)", result.get('family'),
                    result.get('decision_reason'))
        return result

    def classify_fingerprint(self, fingerprint):
        """Apply the same final decision to a collected or replayed fingerprint."""
        try:
            if not isinstance(fingerprint, dict):
                raise FeatureValidationError("Fingerprint must be a record")
            metadata = fingerprint.get('metadata', {})
            if metadata is None:
                metadata = {}
            validate_inference_metadata(metadata)
            vector = validate_fingerprint_vector(fingerprint.get('vector'))
        except (FeatureValidationError, TypeError, ValueError) as exc:
            return {
                'family': 'unknown', 'ood_detected': True,
                'decision_reason': 'invalid_fingerprint',
                'error': f"Invalid fingerprint: {exc}",
                'fingerprint': fingerprint,
            }
        incomplete = bool(metadata.get('incomplete') or metadata.get('layers_skipped'))
        if metadata.get('early_stopped') and len(metadata.get('layers_completed', [])) < len(config.LAYER_ORDER):
            incomplete = True
        pipeline = IdentificationPipeline(self.classifier, self.family_templates,
                                          self.model_templates)
        result = pipeline.classify(vector, incomplete=incomplete)
        result.update({
            'model': fingerprint.get('model'),
            'early_stopped': bool(metadata.get('early_stopped')),
            'layers_completed': metadata.get('layers_completed', []),
            'queries_executed': metadata.get('queries_executed', 0),
            'queries_total': metadata.get('queries_total', 0),
            'fingerprint': fingerprint,
        })
        return result
