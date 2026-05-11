#!/usr/bin/env python3
"""
Analyze which prompts (and prompt categories) are most discriminative.

Works in two modes:

  Training-data mode (default, no API needed):
    Loads stored training fingerprints and computes per-feature Fisher
    discriminability, then aggregates by prompt layer and category to show
    which types of prompts drive the most separation between families.

  Live mode (--live, requires API):
    Fingerprints a target endpoint using each prompt INDIVIDUALLY,
    compares responses to trained templates, and ranks prompts by how
    well each one alone separates the target from other families.
    Useful for pruning the prompt suite or building a fast-path classifier.

Usage:
    # Training-data analysis (fast, no API)
    python analyze_prompts.py

    # Live per-prompt scoring against a running endpoint
    python analyze_prompts.py --live --endpoint http://localhost:11434/ \\
        --model llama3.2 --backend ollama

    # Show only discriminative-layer prompts
    python analyze_prompts.py --layer discriminative
"""
from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from llm_fingerprinter import config
from llm_fingerprinter.fingerprint_store import FingerprintStore
from llm_fingerprinter.prompt_suite import PromptSuite
from llm_fingerprinter.classifier import EnsembleClassifier

logging.basicConfig(level=logging.WARNING)


# ── Discriminability helpers ───────────────────────────────────────────────────

def fisher_ratios(training_data: dict) -> np.ndarray | None:
    """Per-feature Fisher ratio from training fingerprints.

    Returns array of shape (n_features,) or None if < 2 families.
    """
    if len(training_data) < 2:
        return None

    matrices = {f: np.array(v, dtype=np.float32)
                for f, v in training_data.items() if v}
    if len(matrices) < 2:
        return None

    all_vecs   = np.concatenate(list(matrices.values()), axis=0)
    grand_mean = all_vecs.mean(axis=0)
    n_features = all_vecs.shape[1]

    between = np.zeros(n_features, dtype=np.float64)
    within  = np.zeros(n_features, dtype=np.float64)

    for mat in matrices.values():
        cm = mat.mean(axis=0)
        between += len(mat) * (cm - grand_mean) ** 2
        within  += ((mat - cm) ** 2).sum(axis=0)

    between /= len(matrices)
    within  /= max(len(all_vecs) - len(matrices), 1)

    return between / (within + 1e-10)


def pairwise_cosine_distances(training_data: dict) -> tuple[list[str], np.ndarray] | None:
    """Compute pairwise cosine distances between family mean vectors.

    Returns:
        (families, dist_matrix) where dist_matrix[i,j] is the cosine distance
        between family i and family j (0 = identical, 1 = orthogonal, 2 = opposite).
        Lower = more similar = harder to tell apart.
    """
    matrices = {f: np.array(v, dtype=np.float32)
                for f, v in training_data.items() if v}
    if len(matrices) < 2:
        return None

    families = sorted(matrices.keys())
    means = np.stack([matrices[f].mean(axis=0) for f in families])  # (K, D)

    # Normalise rows
    norms = np.linalg.norm(means, axis=1, keepdims=True) + 1e-9
    normed = means / norms

    # Cosine similarity → distance
    sim = normed @ normed.T
    dist = 1.0 - np.clip(sim, -1.0, 1.0)
    return families, dist


# ── Prompt-level aggregation ───────────────────────────────────────────────────

def layer_discriminability(scores: np.ndarray) -> dict[str, float]:
    """Mean Fisher ratio per prompt layer."""
    per_layer = config.PER_PROMPT_FEATURE_DIM
    result = {}
    for i, layer in enumerate(config.LAYER_ORDER):
        start = i * per_layer
        result[layer] = float(scores[start:start + per_layer].mean())
    return result


def _print_bar(label: str, val: float, scale: float, width: int = 40):
    bar_len = max(1, int(val * scale))
    bar = "█" * min(bar_len, width)
    print(f"  {label:<24}  {val:>8.3f}  {bar}")


# ── Live per-prompt mode ───────────────────────────────────────────────────────

def run_live_analysis(args):
    """Run each prompt individually, score vs templates, rank by discriminability."""
    from llm_fingerprinter.template_classifier import TemplateClassifier
    from llm_fingerprinter.feature_extractor import FeatureExtractor

    # ── Build client ───────────────────────────────────────────────────────────
    endpoint = args.endpoint
    backend  = args.backend or config.DEFAULT_BACKEND

    if backend == "ollama":
        from llm_fingerprinter.ollama_client import OllamaClient
        client = OllamaClient(endpoint)
    elif backend == "openai":
        import os
        from llm_fingerprinter.openai_client import OpenAIClient
        client = OpenAIClient(os.environ.get("OPENAI_API_KEY", ""), endpoint=endpoint)
    elif backend == "gemini":
        import os
        from llm_fingerprinter.gemini_client import GeminiClient
        client = GeminiClient(os.environ.get("GEMINI_API_KEY", ""), endpoint=endpoint)
    elif backend == "custom":
        import os
        from llm_fingerprinter.custom_client import CustomClient
        client = CustomClient(api_key=os.environ.get("OPENAI_API_KEY", ""))
    else:
        print(f"Unsupported backend '{backend}'. Use: ollama / openai / gemini / custom")
        sys.exit(1)

    # ── Load templates ─────────────────────────────────────────────────────────
    tc = TemplateClassifier()
    if not tc.load(str(config.TEMPLATES_PATH)):
        print(f"\nERROR: No templates found at {config.TEMPLATES_PATH}")
        print("Run 'llm-fingerprinter build-templates' first.\n")
        sys.exit(1)

    extractor = FeatureExtractor()
    suite     = PromptSuite()
    prompts   = suite.get_prompts()
    if args.layer:
        prompts = [p for p in prompts if p["layer"] == args.layer]

    model  = args.model
    scores = []

    print(f"\nRunning {len(prompts)} prompts against {model} at {endpoint} ...\n")

    for i, pd in enumerate(prompts, 1):
        prompt_text = pd["text"]
        try:
            response = client.generate(
                model=model,
                prompt=prompt_text,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
        except Exception as e:
            print(f"  [{i:>3}/{len(prompts)}] ERROR: {e}")
            scores.append({"prompt": pd, "score": 0.0, "family": "?", "error": True})
            continue

        feat = extractor.extract(prompt_text, response)
        # Use feat as a single-layer fingerprint (partial)
        # Score = template confidence for this single response
        try:
            result = tc.classify(feat)
            confidence = result["confidence"] if not result["is_ood"] else 0.0
        except Exception:
            confidence = 0.0

        scores.append({
            "prompt": pd,
            "score": confidence,
            "family": result.get("predicted_family", "?"),
            "error": False,
        })
        short = prompt_text[:60].replace("\n", " ")
        print(f"  [{i:>3}/{len(prompts)}]  conf={confidence:.3f}  {short}…")

    # ── Report ─────────────────────────────────────────────────────────────────
    scores.sort(key=lambda x: -x["score"])
    print(f"\n{'='*65}")
    print("  Prompt Discriminability Ranking  (live mode)")
    print(f"{'='*65}\n")
    print(f"  {'Rank':<5}  {'Score':>6}  {'Layer':<16}  {'Category':<14}  Prompt")
    print(f"  {'─'*5}  {'─'*6}  {'─'*16}  {'─'*14}  {'─'*40}")
    for rank, item in enumerate(scores, 1):
        pd_   = item["prompt"]
        short = pd_["text"][:45].replace("\n", " ")
        print(f"  {rank:<5}  {item['score']:>6.3f}  "
              f"{pd_['layer']:<16}  {pd_.get('category','?'):<14}  {short}…")
    print()


# ── Training-data analysis ─────────────────────────────────────────────────────

def run_training_analysis(args):
    store         = FingerprintStore(args.training_dir)
    training_data = store.export_for_training()

    if not training_data:
        print(f"\nNo training fingerprints found in {args.training_dir}")
        print("Run 'llm-fingerprinter simulate' then 'llm-fingerprinter train' first.\n")
        sys.exit(1)

    families = sorted(training_data.keys())
    counts   = {f: len(v) for f, v in training_data.items()}

    print(f"\n{'='*65}")
    print("  Prompt Discriminability Analysis  (training-data mode)")
    print(f"{'='*65}")
    print(f"\n  Families: {', '.join(f'{f} ({counts[f]})' for f in families)}")

    scores = fisher_ratios(training_data)
    if scores is None:
        print("\n  Need >= 2 families with fingerprints for Fisher analysis.\n")
        sys.exit(1)

    # ── Layer-level discriminability ───────────────────────────────────────────
    layer_disc = layer_discriminability(scores)
    best_layer = max(layer_disc, key=layer_disc.get)
    scale = 40.0 / max(layer_disc.values()) if max(layer_disc.values()) > 0 else 1.0

    print("\nDiscriminability by prompt layer (mean Fisher ratio):")
    for layer, val in sorted(layer_disc.items(), key=lambda x: -x[1]):
        label = f"  ★ {layer}" if layer == best_layer else f"    {layer}"
        _print_bar(label, val, scale)

    # ── Feature-type breakdown per layer ───────────────────────────────────────
    per_layer = config.PER_PROMPT_FEATURE_DIM
    embed_dim = config.EMBEDDING_DIM
    ling_dim  = config.LINGUISTIC_DIM
    behav_dim = config.BEHAVIORAL_DIM

    # Compute all values first so we can scale bars relative to the global max
    layer_type_vals = []
    for i, layer in enumerate(config.LAYER_ORDER):
        start  = i * per_layer
        e_mean = float(scores[start:start + embed_dim].mean())
        l_mean = float(scores[start + embed_dim:start + embed_dim + ling_dim].mean())
        b_mean = float(scores[start + embed_dim + ling_dim:
                                start + embed_dim + ling_dim + behav_dim].mean())
        layer_type_vals.append((layer, e_mean, l_mean, b_mean))

    global_max = max(v for _, e, l, b in layer_type_vals for v in (e, l, b))
    type_scale = 40.0 / global_max if global_max > 0 else 1.0

    print("\nDiscriminability by feature type within each layer:")
    for layer, e_mean, l_mean, b_mean in layer_type_vals:
        print(f"\n  {layer}:")
        _print_bar("    embedding",  e_mean, type_scale)
        _print_bar("    linguistic", l_mean, type_scale)
        _print_bar("    behavioral", b_mean, type_scale)

    # ── Top discriminative linguistic / behavioral features ────────────────────
    LINGUISTIC_NAMES = [
        "total_chars", "total_words", "type_token_ratio", "avg_word_len",
        "num_sentences", "avg_sent_len", "punctuation_ratio", "code_block_ratio",
        "structural_markers_ratio", "token_entropy", "ai_marker_count",
        "capital_ratio",
    ]
    BEHAVIORAL_NAMES = [
        "refusal_score", "format_adherence", "reasoning_presence",
        "instruction_compliance", "length_normalization", "formality_score",
    ]

    ling_agg: dict[str, float] = {}
    beh_agg:  dict[str, float] = {}

    for dim, score in enumerate(scores):
        slot = dim % per_layer
        if embed_dim <= slot < embed_dim + ling_dim:
            name = LINGUISTIC_NAMES[slot - embed_dim]
            ling_agg[name] = ling_agg.get(name, 0.0) + score
        elif slot >= embed_dim + ling_dim:
            name = BEHAVIORAL_NAMES[slot - embed_dim - ling_dim]
            beh_agg[name] = beh_agg.get(name, 0.0) + score

    print("\nTop linguistic signals (summed Fisher across all layers):")
    max_l = max(ling_agg.values()) if ling_agg else 1.0
    for name, val in sorted(ling_agg.items(), key=lambda x: -x[1]):
        _print_bar(f"  {name}", val, 20.0 / max_l)

    print("\nTop behavioral signals (summed Fisher across all layers):")
    max_b = max(beh_agg.values()) if beh_agg else 1.0
    for name, val in sorted(beh_agg.items(), key=lambda x: -x[1]):
        _print_bar(f"  {name}", val, 20.0 / max_b)

    # ── Pairwise family similarity matrix ─────────────────────────────────────
    pair_result = pairwise_cosine_distances(training_data)
    if pair_result:
        fams, dist_matrix = pair_result
        K = len(fams)

        print(f"\n{'='*65}")
        print("  Pairwise Family Similarity  (cosine distance between means)")
        print("  Lower = more similar = harder to tell apart")
        print(f"{'='*65}\n")

        # Header row
        col_w = 10
        header = f"  {'':16}" + "".join(f"{f:>{col_w}}" for f in fams)
        print(header)
        print("  " + "─" * (16 + col_w * K))

        for i, fi in enumerate(fams):
            row = f"  {fi:<16}"
            for j, fj in enumerate(fams):
                if i == j:
                    row += f"{'—':>{col_w}}"
                else:
                    d = dist_matrix[i, j]
                    row += f"{d:>{col_w}.4f}"
            print(row)

        # Highlight the most and least separable pairs
        pairs = [(dist_matrix[i, j], fams[i], fams[j])
                 for i in range(K) for j in range(i + 1, K)]
        pairs.sort()

        print(f"\n  Hardest to distinguish (most similar):")
        for d, fa, fb in pairs[:3]:
            bar = "█" * max(1, int((1 - d) * 30))
            print(f"    {fa} ↔ {fb:<16}  dist={d:.4f}  {bar}")

        print(f"\n  Easiest to distinguish (most different):")
        for d, fa, fb in reversed(pairs[-3:]):
            bar = "█" * max(1, int(d * 30))
            print(f"    {fa} ↔ {fb:<16}  dist={d:.4f}  {bar}")

    # ── Prompt listing by layer importance ────────────────────────────────────
    suite   = PromptSuite()
    all_p   = suite.get_prompts()
    filter_ = args.layer

    print(f"\n{'='*65}")
    print("  Prompts listed by layer (most discriminative layer first)")
    print(f"{'='*65}")

    for layer in sorted(config.LAYER_ORDER, key=lambda l: -layer_disc[l]):
        if filter_ and layer != filter_:
            continue
        prompts_in_layer = [p for p in all_p if p["layer"] == layer]
        # Group by category
        by_cat: dict[str, list] = {}
        for p in prompts_in_layer:
            cat = p.get("category", "other")
            by_cat.setdefault(cat, []).append(p)

        print(f"\n  [{layer.upper()}]  (Fisher={layer_disc[layer]:.3f})")
        for cat, cat_prompts in sorted(by_cat.items()):
            print(f"\n    Category: {cat} ({len(cat_prompts)} prompts)")
            for p in cat_prompts:
                short = p["text"][:80].replace("\n", " ")
                print(f"      • {short}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze which prompts are most discriminative for LLM fingerprinting."
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run each prompt individually against a live API endpoint"
    )
    parser.add_argument(
        "--endpoint",
        help="API endpoint URL (required for --live)"
    )
    parser.add_argument(
        "--model",
        help="Model name on the endpoint (required for --live)"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai", "gemini", "custom"],
        help="API backend (default: custom)"
    )
    parser.add_argument(
        "--layer", choices=config.LAYER_ORDER,
        help="Filter analysis to one prompt layer"
    )
    parser.add_argument(
        "--training-dir",
        default=str(config.TRAINING_DIR),
        help="Training fingerprints directory"
    )
    args = parser.parse_args()

    if args.live:
        if not args.endpoint or not args.model:
            parser.error("--live requires --endpoint and --model")
        run_live_analysis(args)
    else:
        run_training_analysis(args)


if __name__ == "__main__":
    main()
