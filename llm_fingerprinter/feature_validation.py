"""Contracts shared by response extraction, collection and training imports."""

from collections.abc import Mapping
import hashlib
import json

import numpy as np

from llm_fingerprinter import config


class FeatureValidationError(ValueError):
    """Feature data cannot safely be used for collection or classification."""


def _finite_vector(value, size, name):
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureValidationError(f"{name} must contain numeric features") from exc
    if array.shape != (size,):
        raise FeatureValidationError(f"{name} has shape {array.shape}; expected ({size},)")
    if not np.all(np.isfinite(array)):
        raise FeatureValidationError(f"{name} contains nonfinite values")
    return array


def validate_embedding(embedding, embedding_dim=config.EMBEDDING_DIM):
    array = _finite_vector(embedding, embedding_dim, "Embedding")
    if not np.any(array):
        raise FeatureValidationError("Embedding is all zero; feature extraction failed")
    return array


def validate_response_features(vector, *, embedding_dim=config.EMBEDDING_DIM,
                               linguistic_dim=config.LINGUISTIC_DIM,
                               behavioral_dim=config.BEHAVIORAL_DIM):
    size = embedding_dim + linguistic_dim + behavioral_dim
    array = _finite_vector(vector, size, "Response features")
    validate_embedding(array[:embedding_dim], embedding_dim)
    return array


def validate_fingerprint_vector(vector, *, embedding_dim=config.EMBEDDING_DIM,
                                linguistic_dim=config.LINGUISTIC_DIM,
                                behavioral_dim=config.BEHAVIORAL_DIM,
                                layer_order=None):
    layers = config.LAYER_ORDER if layer_order is None else layer_order
    size = embedding_dim + linguistic_dim + behavioral_dim
    array = _finite_vector(vector, len(layers) * size, "Fingerprint")
    for index, layer in enumerate(layers):
        try:
            validate_embedding(array[index * size:index * size + embedding_dim], embedding_dim)
        except FeatureValidationError as exc:
            raise FeatureValidationError(f"Layer '{layer}': {exc}") from exc
    return array


def feature_schema(embedding_model=config.EMBEDDING_MODEL, *,
                   embedding_dim=config.EMBEDDING_DIM,
                   linguistic_dim=config.LINGUISTIC_DIM,
                   behavioral_dim=config.BEHAVIORAL_DIM, layer_order=None):
    """Describe the representation so incompatible future exports fail visibly."""
    return {
        "version": 1,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "linguistic_dim": linguistic_dim,
        "behavioral_dim": behavioral_dim,
        "layer_order": list(config.LAYER_ORDER if layer_order is None else layer_order),
        "aggregation": "layer_mean",
    }


def current_prompt_suite_hash(prompts=None):
    """Hash the ordered prompts and their layer/category assignments."""
    if prompts is None:
        from llm_fingerprinter.prompt_suite import PromptSuite
        prompts = PromptSuite().get_prompts()
    payload = json.dumps(prompts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_training_metadata(metadata):
    """Require complete observations and the stricter training coverage gate."""
    _validate_metadata(metadata, config.TRAINING_MIN_LAYER_COVERAGE, allow_partial=False)


def validate_inference_metadata(metadata):
    """Validate replay provenance, permitting documented deliberate early stops.

    Partial observations remain unsuitable for a verified identification. This
    check allows the inference policy to return unknown for deliberate early
    stopping while treating corrupted or failed collection records as errors.
    """
    _validate_metadata(metadata, config.MIN_LAYER_COVERAGE, allow_partial=True)


def _validate_metadata(metadata, minimum_coverage, *, allow_partial):
    # Legacy records cannot prove unrecorded provenance, but are accepted when
    # all checks possible from their documented observations pass.
    if metadata is None:
        return
    if not isinstance(metadata, Mapping):
        raise FeatureValidationError("Fingerprint metadata must be an object")

    schema = metadata.get("feature_schema")
    if "feature_schema" in metadata:
        if not isinstance(schema, Mapping):
            raise FeatureValidationError("Feature schema must be an object")
        for key, expected in feature_schema().items():
            if schema.get(key) != expected:
                raise FeatureValidationError(
                    f"Incompatible feature schema '{key}': {schema.get(key)!r}; expected {expected!r}"
                )
    if ("prompt_suite_hash" in metadata
            and metadata["prompt_suite_hash"] != current_prompt_suite_hash()):
        raise FeatureValidationError("Fingerprint uses a different prompt suite")
    if "feature_dim" in metadata and metadata["feature_dim"] != config.RAW_FINGERPRINT_DIM:
        raise FeatureValidationError("Metadata feature_dim does not match the feature schema")

    if metadata.get("status") in ("failed", "error"):
        raise FeatureValidationError("Fingerprint collection did not succeed")
    for flag in ("layers_failed", "error", "collection_failed", "feature_extraction_failed"):
        if metadata.get(flag):
            raise FeatureValidationError(f"Fingerprint metadata records {flag}={metadata[flag]!r}")
    for flag in ("quality_passed", "collection_success", "feature_extraction_success"):
        if flag in metadata and metadata[flag] is not True:
            raise FeatureValidationError(f"Fingerprint metadata records {flag}={metadata[flag]!r}")
    for flag in ("early_stopped", "incomplete"):
        if flag in metadata and not isinstance(metadata[flag], bool):
            raise FeatureValidationError(f"Metadata {flag} must be a boolean")

    expected_layers = set(config.LAYER_ORDER)

    def layer_set(name, default):
        if name not in metadata:
            return default
        layers = metadata[name]
        if (not isinstance(layers, (list, tuple))
                or not all(isinstance(layer, str) for layer in layers)
                or len(layers) != len(set(layers))
                or not set(layers).issubset(expected_layers)):
            raise FeatureValidationError(f"Metadata {name} must list unique known prompt layers")
        return set(layers)

    completed = layer_set("layers_completed", expected_layers)
    skipped = layer_set("layers_skipped", set())
    partial = bool(metadata.get("incomplete") or skipped or completed != expected_layers)
    if partial:
        if not (allow_partial and metadata.get("early_stopped") is True
                and metadata.get("incomplete") is True
                and "layers_completed" in metadata and completed and skipped
                and not completed.intersection(skipped)
                and completed.union(skipped) == expected_layers):
            raise FeatureValidationError("Missing layers are not a documented deliberate early stop")
    elif metadata.get("early_stopped") and "layers_completed" not in metadata:
        raise FeatureValidationError("Early-stopped data does not document its completed layers")
    if metadata.get("status") == "incomplete" and not partial:
        raise FeatureValidationError("Fingerprint collection is marked incomplete")
    if "layers_attempted" in metadata and layer_set("layers_attempted", set()) != completed:
        raise FeatureValidationError("Attempted and completed layers disagree")

    def number(value, name, *, maximum=None):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise FeatureValidationError(f"Metadata {name} must be numeric")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0 or (maximum is not None and numeric > maximum):
            raise FeatureValidationError(f"Invalid metadata {name}: {value!r}")
        return numeric

    def count(value, name, *, positive=False):
        numeric = number(value, name)
        if numeric != int(numeric) or (positive and numeric < 1):
            raise FeatureValidationError(f"Metadata {name} must be a {'positive ' if positive else ''}integer count")
        return numeric

    for key in ("queries_executed", "queries_total", "queries_attempted", "queries_failed", "queries_empty"):
        if key in metadata:
            count(metadata[key], key, positive=key in ("queries_executed", "queries_total", "queries_attempted"))
    for key in ("repeats", "prompt_count", "max_tokens"):
        if key in metadata:
            count(metadata[key], key, positive=True)
    for key in ("temperature", "duration_seconds"):
        if key in metadata:
            number(metadata[key], key)
    if "min_layer_coverage" in metadata:
        recorded_minimum = number(metadata["min_layer_coverage"], "min_layer_coverage", maximum=1)
        if recorded_minimum == 0:
            raise FeatureValidationError("Metadata min_layer_coverage must be positive")
        minimum_coverage = max(minimum_coverage, recorded_minimum)

    layers = metadata.get("layers")
    expected_coverage = {}
    if "layers" in metadata:
        if not isinstance(layers, Mapping) or set(layers) not in (completed, expected_layers):
            raise FeatureValidationError("Missing layer observation counts")
        for layer, value in layers.items():
            observed = count(value, f"layers.{layer}", positive=layer in completed)
            if layer in skipped and observed != 0:
                raise FeatureValidationError(f"Skipped layer '{layer}' records observations")
        if "queries_executed" in metadata and sum(layers.values()) != metadata["queries_executed"]:
            raise FeatureValidationError("Layer observation counts disagree with queries_executed")
        if "repeats" in metadata:
            from llm_fingerprinter.prompt_suite import PromptSuite
            suite = PromptSuite()
            if metadata.get("prompt_count") == len(suite) or "prompt_suite_hash" in metadata:
                for layer in completed:
                    expected = len(suite.get_prompts(layer=layer)) * metadata["repeats"]
                    expected_coverage[layer] = layers[layer] / expected
                    if not (minimum_coverage <= expected_coverage[layer] <= 1):
                        raise FeatureValidationError(f"Layer '{layer}' counts fail the collection coverage minimum")

    if "layer_coverage" in metadata:
        coverage = metadata["layer_coverage"]
        if not isinstance(coverage, Mapping) or set(coverage) not in (completed, expected_layers):
            raise FeatureValidationError("Missing per-layer coverage")
        for layer, value in coverage.items():
            ratio = number(value, f"layer_coverage.{layer}", maximum=1)
            if layer in completed and ratio < minimum_coverage:
                raise FeatureValidationError(f"Layer '{layer}' is below the collection coverage minimum")
            if layer in skipped and ratio != 0:
                raise FeatureValidationError(f"Skipped layer '{layer}' records coverage")
            if layer in expected_coverage and abs(ratio - expected_coverage[layer]) > 0.000500001:
                raise FeatureValidationError(f"Layer '{layer}' coverage disagrees with observation counts")

    for key in ("success_rate", "suite_coverage"):
        if key in metadata:
            ratio = number(metadata[key], key, maximum=1)
            if not (partial and key == "suite_coverage") and ratio < minimum_coverage:
                raise FeatureValidationError(f"Metadata {key} is below the collection coverage minimum")
    if "queries_executed" in metadata and "queries_total" in metadata:
        ratio = metadata["queries_executed"] / metadata["queries_total"]
        if ratio > 1 or (not partial and ratio < minimum_coverage):
            raise FeatureValidationError("Recorded query counts do not meet collection coverage")
    for smaller, larger in (("queries_executed", "queries_attempted"),
                            ("queries_attempted", "queries_total"),
                            ("queries_empty", "queries_failed"),
                            ("queries_failed", "queries_attempted")):
        if smaller in metadata and larger in metadata and metadata[smaller] > metadata[larger]:
            raise FeatureValidationError(f"Metadata {smaller} exceeds {larger}")
    if all(key in metadata for key in ("queries_attempted", "queries_executed", "queries_failed")):
        if metadata["queries_attempted"] != metadata["queries_executed"] + metadata["queries_failed"]:
            raise FeatureValidationError("Attempted, successful and failed query counts disagree")
    for rate, denominator in (("success_rate", "queries_attempted"), ("suite_coverage", "queries_total")):
        if all(key in metadata for key in (rate, "queries_executed", denominator)):
            expected = metadata["queries_executed"] / metadata[denominator]
            # Collection persists these rates rounded to three decimal places.
            if abs(metadata[rate] - expected) > 0.000500001:
                raise FeatureValidationError(f"Metadata {rate} disagrees with recorded query counts")
