#!/usr/bin/env python3
"""
Analyze which feature dimensions are most discriminative for LLM classification.

Uses the trained Random Forest classifier's feature importances to rank features,
then maps them back to their semantic meaning:

  Layer (discriminative / behavioral / stylistic)
  × Feature type (embedding / linguistic / behavioral)
  × Named feature (for linguistic and behavioral slots)

Usage:
    python analyze_features.py                       # top 30 features
    python analyze_features.py --top 50              # top 50 features
    python analyze_features.py --layer discriminative
    python analyze_features.py --type linguistic
    python analyze_features.py --layer behavioral --type behavioral
"""
from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

# Ensure repo root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent))

from llm_fingerprinter import config
from llm_fingerprinter.fingerprint_store import FingerprintStore
from llm_fingerprinter.classifier import EnsembleClassifier

logging.basicConfig(level=logging.WARNING)

# ── Semantic feature name maps ─────────────────────────────────────────────────
LINGUISTIC_NAMES = [
    "total_chars",
    "total_words",
    "type_token_ratio",
    "avg_word_len",
    "num_sentences",
    "avg_sent_len",
    "punctuation_ratio",
    "code_block_ratio",
    "structural_markers_ratio",
    "token_entropy",
    "ai_marker_count",
    "capital_ratio",
]

BEHAVIORAL_NAMES = [
    "refusal_score",
    "format_adherence",
    "reasoning_presence",
    "instruction_compliance",
    "length_normalization",
    "formality_score",
]


def _get_layout(n_features: int) -> tuple[int, int, int]:
    """Return (per_layer, embed_dim, ling_dim) for either raw or PCA-rebalanced space.

    Raw (1206):        3 × (384 embed + 12 ling + 6 beh)
    Rebalanced (246):  3 × (64 PCA-embed + 12 ling + 6 beh)
    Any other size:    infer embed_dim from per_layer - 18
    """
    n_layers  = config.NUM_PROMPT_LAYERS       # 3
    ling_dim  = config.LINGUISTIC_DIM          # 12
    behav_dim = config.BEHAVIORAL_DIM          #  6
    non_embed = ling_dim + behav_dim           # 18

    if n_features == config.RAW_FINGERPRINT_DIM:
        # Raw un-compressed features
        return config.PER_PROMPT_FEATURE_DIM, config.EMBEDDING_DIM, ling_dim

    # PCA-rebalanced (or any other compressed form)
    per_layer = n_features // n_layers
    embed_dim = per_layer - non_embed
    return per_layer, embed_dim, ling_dim


def feature_name(idx: int, n_features: int) -> str:
    """Return a human-readable name for feature dimension idx (0-based)."""
    per_layer, embed_dim, ling_dim = _get_layout(n_features)
    behav_dim = config.BEHAVIORAL_DIM

    layer_idx = idx // per_layer
    slot      = idx %  per_layer

    layer = (config.LAYER_ORDER[layer_idx]
             if layer_idx < len(config.LAYER_ORDER)
             else f"layer{layer_idx}")

    if slot < embed_dim:
        return f"{layer}/embedding_pca[{slot}]" if n_features != config.RAW_FINGERPRINT_DIM else f"{layer}/embedding[{slot}]"
    slot -= embed_dim
    if slot < ling_dim:
        return f"{layer}/linguistic/{LINGUISTIC_NAMES[slot]}"
    slot -= ling_dim
    if slot < behav_dim:
        return f"{layer}/behavioral/{BEHAVIORAL_NAMES[slot]}"
    return f"{layer}/unknown[{slot}]"


def feature_type(idx: int, n_features: int) -> str:
    per_layer, embed_dim, ling_dim = _get_layout(n_features)
    slot = idx % per_layer
    if slot < embed_dim:
        return "embedding"
    if slot < embed_dim + ling_dim:
        return "linguistic"
    return "behavioral"


def feature_layer(idx: int, n_features: int) -> str:
    per_layer, _, _ = _get_layout(n_features)
    layer_idx = idx // per_layer
    if layer_idx < len(config.LAYER_ORDER):
        return config.LAYER_ORDER[layer_idx]
    return f"layer{layer_idx}"


# ── Fisher ratio from training data ───────────────────────────────────────────

def compute_fisher_ratios(store_path: str) -> tuple[np.ndarray, list[str]] | None:
    """Compute per-feature Fisher discriminability from training fingerprints.

    Fisher ratio = between_class_variance / within_class_variance (per feature).
    Higher = more discriminative.

    Returns:
        (fisher_ratios, family_labels) or None on failure.
    """
    store = FingerprintStore(store_path)
    training_data = store.export_for_training()

    if len(training_data) < 2:
        return None

    families = sorted(training_data.keys())
    all_matrices = {}
    for fam, vecs in training_data.items():
        if vecs:
            all_matrices[fam] = np.array(vecs, dtype=np.float32)

    if len(all_matrices) < 2:
        return None

    # Global mean
    all_vecs = np.concatenate(list(all_matrices.values()), axis=0)
    grand_mean = all_vecs.mean(axis=0)

    n_features = all_vecs.shape[1]
    between_var = np.zeros(n_features, dtype=np.float64)
    within_var  = np.zeros(n_features, dtype=np.float64)

    for fam, mat in all_matrices.items():
        class_mean = mat.mean(axis=0)
        n = len(mat)
        between_var += n * (class_mean - grand_mean) ** 2
        within_var  += ((mat - class_mean) ** 2).sum(axis=0)

    between_var /= len(all_matrices)
    within_var  /= max(len(all_vecs) - len(all_matrices), 1)

    fisher = between_var / (within_var + 1e-10)
    return fisher, families


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Show which features matter most for LLM fingerprint classification."
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of top features to display (default: 30)"
    )
    parser.add_argument(
        "--layer", choices=config.LAYER_ORDER,
        help="Filter to a specific prompt layer"
    )
    parser.add_argument(
        "--type", dest="feat_type",
        choices=["embedding", "linguistic", "behavioral"],
        help="Filter to a specific feature type"
    )
    parser.add_argument(
        "--model-path",
        default=str(config.MODEL_DIR / "classifier_model.joblib"),
        help="Path to trained classifier joblib"
    )
    parser.add_argument(
        "--fisher", action="store_true",
        help="Also show Fisher ratio from training data (slower)"
    )
    parser.add_argument(
        "--training-dir",
        default=str(config.TRAINING_DIR),
        help="Training fingerprints directory (for --fisher)"
    )
    args = parser.parse_args()

    # ── Load classifier ────────────────────────────────────────────────────────
    clf = EnsembleClassifier()
    if not clf.load(args.model_path):
        print(f"\nERROR: Could not load classifier from {args.model_path}")
        print("Run 'llm-fingerprinter train' first.\n")
        sys.exit(1)

    if clf.rf is None or not hasattr(clf.rf, "feature_importances_"):
        print("\nERROR: RF classifier not available or not fitted.\n")
        sys.exit(1)

    importances = clf.rf.feature_importances_
    n_features  = len(importances)
    per_layer, embed_dim, ling_dim = _get_layout(n_features)
    behav_dim   = config.BEHAVIORAL_DIM
    space_label = "raw" if n_features == config.RAW_FINGERPRINT_DIM else f"PCA-rebalanced"

    print(f"\n{'='*65}")
    print(f"  RF Feature Importance Analysis  —  {n_features} dims ({space_label})")
    print(f"{'='*65}")

    # ── Per-layer aggregate ────────────────────────────────────────────────────
    print("\nImportance by prompt layer (higher = layer's prompts matter more):")
    for i, layer in enumerate(config.LAYER_ORDER):
        start = i * per_layer
        val   = importances[start:start + per_layer].sum()
        bar   = "█" * max(1, int(val * 300))
        print(f"  {layer:<16}  {val:.4f}  {bar}")

    # ── Per-type aggregate ─────────────────────────────────────────────────────
    totals = {"embedding": 0.0, "linguistic": 0.0, "behavioral": 0.0}
    for i, imp in enumerate(importances):
        totals[feature_type(i, n_features)] += imp

    print("\nImportance by feature type:")
    for label, val in totals.items():
        bar = "█" * max(1, int(val * 300))
        print(f"  {label:<16}  {val:.4f}  {bar}")

    # ── Named linguistic / behavioral breakdown ────────────────────────────────
    print("\nNamed linguistic features (summed across layers):")
    ling_by_name: dict[str, float] = {}
    for i, imp in enumerate(importances):
        slot = i % per_layer
        if embed_dim <= slot < embed_dim + ling_dim:
            name = LINGUISTIC_NAMES[slot - embed_dim]
            ling_by_name[name] = ling_by_name.get(name, 0.0) + imp

    if ling_by_name:
        for name, val in sorted(ling_by_name.items(), key=lambda x: -x[1]):
            bar = "█" * max(1, int(val * 3000))
            print(f"  {name:<32}  {val:.6f}  {bar}")
    else:
        print("  (none — all importance in embedding dims)")

    print("\nNamed behavioral features (summed across layers):")
    beh_by_name: dict[str, float] = {}
    for i, imp in enumerate(importances):
        slot = i % per_layer
        if embed_dim + ling_dim <= slot < embed_dim + ling_dim + behav_dim:
            name = BEHAVIORAL_NAMES[slot - embed_dim - ling_dim]
            beh_by_name[name] = beh_by_name.get(name, 0.0) + imp

    if beh_by_name:
        for name, val in sorted(beh_by_name.items(), key=lambda x: -x[1]):
            bar = "█" * max(1, int(val * 3000))
            print(f"  {name:<32}  {val:.6f}  {bar}")
    else:
        print("  (none — all importance in embedding dims)")

    # ── Top individual features ────────────────────────────────────────────────
    indices = list(np.argsort(importances)[::-1])

    if args.layer or args.feat_type:
        indices = [
            idx for idx in indices
            if (not args.layer     or feature_layer(idx, n_features) == args.layer)
            and (not args.feat_type or feature_type(idx, n_features) == args.feat_type)
        ]

    top_n = min(args.top, len(indices))
    print(f"\nTop {top_n} individual feature dimensions:")
    print(f"  {'Rank':<5}  {'Importance':>10}  Feature")
    print(f"  {'─'*5}  {'─'*10}  {'─'*55}")
    cumulative = 0.0
    for rank, idx in enumerate(indices[:top_n], 1):
        imp = importances[idx]
        cumulative += imp
        print(f"  {rank:<5}  {imp:>10.6f}  {feature_name(idx, n_features)}")

    coverage = cumulative / importances.sum() if importances.sum() > 0 else 0
    print(f"\n  Top {top_n} features cover {coverage:.1%} of total RF importance.\n")

    # ── Optional Fisher ratio from training data ───────────────────────────────
    if args.fisher:
        print(f"{'='*65}")
        print("  Fisher Discriminability from Training Data")
        print(f"{'='*65}\n")

        result = compute_fisher_ratios(args.training_dir)
        if result is None:
            print("  Not enough training data for Fisher analysis "
                  "(need >= 2 families with fingerprints).\n")
        else:
            fisher, families = result
            f_features = len(fisher)
            print(f"  Families: {', '.join(families)}\n")
            f_per_layer, _, _ = _get_layout(f_features)
            print("  Fisher ratio by prompt layer (higher = more separable):")
            for i, layer in enumerate(config.LAYER_ORDER):
                start = i * f_per_layer
                val   = fisher[start:start + f_per_layer].mean()
                print(f"    {layer:<16}  {val:.4f}")

            print("\n  Top 20 most discriminative dimensions (Fisher ratio):")
            f_indices = np.argsort(fisher)[::-1][:20]
            for rank, idx in enumerate(f_indices, 1):
                print(f"    {rank:>2}.  {fisher[idx]:>8.3f}  {feature_name(idx, f_features)}")
            print()


if __name__ == "__main__":
    main()
