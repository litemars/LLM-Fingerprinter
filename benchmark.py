#!/usr/bin/env python3
"""
Benchmark LLM fingerprinting strategies on known models.

Compares four identification strategies:
  1. full-suite      — all prompts, no early stopping (baseline)
  2. early-stop-90   — stop after a layer if confidence >= 0.90
  3. early-stop-85   — stop after a layer if confidence >= 0.85
  4. template-only   — cosine distance to family templates (no ensemble)

For each strategy, measures:
  - Correct identification rate (accuracy)
  - Average queries used per identification
  - Average wall-clock time per identification

Test models are specified in a JSON config file (see --models-file).
Example models file (benchmark_models.json):
  [
    {"name": "gpt-4o-mini", "family": "gpt",    "endpoint": "https://api.openai.com/v1", "backend": "openai"},
    {"name": "gemma3:latest","family": "gemma",  "endpoint": "http://localhost:11434/",   "backend": "ollama"},
    {"name": "llama3.2",     "family": "llama",  "endpoint": "http://localhost:11434/",   "backend": "ollama"}
  ]

Usage:
    python benchmark.py --models-file benchmark_models.json
    python benchmark.py --models-file benchmark_models.json --repeats 2
    python benchmark.py --models-file benchmark_models.json --strategies full early-stop-90
"""
from __future__ import annotations

import sys
import json
import time
import argparse
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from llm_fingerprinter import config
from llm_fingerprinter.prompt_suite import PromptSuite
from llm_fingerprinter.feature_extractor import FeatureExtractor
from llm_fingerprinter.fingerprint_store import FingerprintStore
from llm_fingerprinter.fingerprinter import LLMFingerprinter
from llm_fingerprinter.classifier import EnsembleClassifier
from llm_fingerprinter.template_classifier import TemplateClassifier

logging.basicConfig(level=logging.WARNING)

STRATEGIES = ["full-suite", "early-stop-90", "early-stop-85", "template-only"]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    family: str
    endpoint: str
    backend: str = "custom"
    api_key: Optional[str] = None


@dataclass
class StrategyResult:
    strategy: str
    model_name: str
    true_family: str
    predicted_family: str
    correct: bool
    confidence: float
    queries_used: int
    queries_total: int
    elapsed_seconds: float
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    strategy: str
    n_models: int
    accuracy: float
    avg_queries: float
    avg_query_pct: float   # fraction of full suite used
    avg_time: float
    results: list[StrategyResult] = field(default_factory=list)


# ── Client factory ─────────────────────────────────────────────────────────────

def build_client(endpoint: str, backend: str, api_key: Optional[str] = None):
    """Build an API client. api_key from the model spec takes priority over env vars."""
    if backend == "ollama":
        from llm_fingerprinter.ollama_client import OllamaClient
        return OllamaClient(endpoint)
    elif backend == "ollama-cloud":
        from llm_fingerprinter.ollama_cloud_client import OllamaCloudClient
        key = api_key or os.environ.get("OLLAMA_CLOUD_API_KEY", "")
        return OllamaCloudClient(key, endpoint=endpoint)
    elif backend == "openai":
        from llm_fingerprinter.openai_client import OpenAIClient
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        return OpenAIClient(key, endpoint=endpoint)
    elif backend == "deepseek":
        from llm_fingerprinter.deepseek_client import DeepSeekClient
        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        return DeepSeekClient(key, endpoint=endpoint)
    elif backend == "gemini":
        from llm_fingerprinter.gemini_client import GeminiClient
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        return GeminiClient(key, endpoint=endpoint)
    else:
        from llm_fingerprinter.custom_client import CustomClient
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        return CustomClient(api_key=key)


# ── Strategy runners ───────────────────────────────────────────────────────────

def run_ensemble_strategy(
    spec: ModelSpec,
    fingerprinter: LLMFingerprinter,
    classifier: EnsembleClassifier,
    early_stop: Optional[float],
    repeats: int,
) -> StrategyResult:
    label = "full-suite" if early_stop is None else f"early-stop-{int(early_stop*100)}"
    t0 = time.time()

    try:
        result = fingerprinter.identify(
            model_name=spec.name,
            repeats=repeats,
            early_stop_confidence=early_stop,
        )
        elapsed = time.time() - t0

        if "error" in result:
            return StrategyResult(
                strategy=label, model_name=spec.name, true_family=spec.family,
                predicted_family="?", correct=False, confidence=0.0,
                queries_used=0, queries_total=0, elapsed_seconds=elapsed,
                error=result["error"],
            )

        predicted = result.get("family", "unknown")
        correct   = predicted == spec.family
        return StrategyResult(
            strategy=label,
            model_name=spec.name,
            true_family=spec.family,
            predicted_family=predicted,
            correct=correct,
            confidence=result.get("confidence", 0.0),
            queries_used=result.get("queries_executed", 0),
            queries_total=result.get("queries_total", 0),
            elapsed_seconds=elapsed,
        )
    except Exception as e:
        return StrategyResult(
            strategy=label, model_name=spec.name, true_family=spec.family,
            predicted_family="?", correct=False, confidence=0.0,
            queries_used=0, queries_total=0,
            elapsed_seconds=time.time() - t0,
            error=str(e),
        )


def run_template_strategy(
    spec: ModelSpec,
    fingerprinter: LLMFingerprinter,
    tc: TemplateClassifier,
    repeats: int,
) -> StrategyResult:
    t0 = time.time()
    try:
        fp = fingerprinter.fingerprint_model(spec.name, repeats=repeats)
        elapsed = time.time() - t0

        if fp is None:
            return StrategyResult(
                strategy="template-only", model_name=spec.name,
                true_family=spec.family, predicted_family="?",
                correct=False, confidence=0.0,
                queries_used=0, queries_total=0, elapsed_seconds=elapsed,
                error="fingerprinting failed",
            )

        result    = tc.classify(fp["vector"])
        predicted = result.get("family", "unknown")
        correct   = predicted == spec.family

        return StrategyResult(
            strategy="template-only",
            model_name=spec.name,
            true_family=spec.family,
            predicted_family=predicted,
            correct=correct,
            confidence=result.get("confidence", 0.0),
            queries_used=fp["metadata"].get("queries_executed", 0),
            queries_total=fp["metadata"].get("queries_total", 0),
            elapsed_seconds=elapsed,
        )
    except Exception as e:
        return StrategyResult(
            strategy="template-only", model_name=spec.name,
            true_family=spec.family, predicted_family="?",
            correct=False, confidence=0.0,
            queries_used=0, queries_total=0,
            elapsed_seconds=time.time() - t0,
            error=str(e),
        )


# ── Summary builder ────────────────────────────────────────────────────────────

def summarize(strategy: str, results: list[StrategyResult]) -> BenchmarkSummary:
    valid = [r for r in results if not r.error]
    n = len(valid)
    if n == 0:
        return BenchmarkSummary(
            strategy=strategy, n_models=len(results),
            accuracy=0.0, avg_queries=0.0, avg_query_pct=0.0, avg_time=0.0,
            results=results,
        )
    accuracy    = sum(r.correct for r in valid) / n
    avg_queries = sum(r.queries_used for r in valid) / n
    pcts = [r.queries_used / r.queries_total
            for r in valid if r.queries_total > 0]
    avg_pct  = sum(pcts) / len(pcts) if pcts else 0.0
    avg_time = sum(r.elapsed_seconds for r in valid) / n
    return BenchmarkSummary(
        strategy=strategy, n_models=len(results),
        accuracy=accuracy, avg_queries=avg_queries,
        avg_query_pct=avg_pct, avg_time=avg_time,
        results=results,
    )


# ── Report printer ─────────────────────────────────────────────────────────────

def print_report(summaries: list[BenchmarkSummary]):
    print(f"\n{'='*75}")
    print("  Benchmark Results")
    print(f"{'='*75}\n")

    # ── Per-model detail ───────────────────────────────────────────────────────
    all_models = sorted({r.model_name
                         for s in summaries
                         for r in s.results})

    print(f"  {'Model':<20}  {'Family':<10}", end="")
    for s in summaries:
        short = s.strategy.replace("early-stop-", "es@").replace("full-suite", "full")
        print(f"  {short:<14}", end="")
    print()
    print(f"  {'─'*20}  {'─'*10}", end="")
    for _ in summaries:
        print(f"  {'─'*14}", end="")
    print()

    for model in all_models:
        # Get true family from first strategy's result
        true_fam = "?"
        for s in summaries:
            for r in s.results:
                if r.model_name == model:
                    true_fam = r.true_family
                    break
            if true_fam != "?":
                break

        print(f"  {model:<20}  {true_fam:<10}", end="")
        for s in summaries:
            r_match = next((r for r in s.results if r.model_name == model), None)
            if r_match is None:
                print(f"  {'–':<14}", end="")
            elif r_match.error:
                print(f"  {'ERROR':<14}", end="")
            else:
                tick = "✓" if r_match.correct else "✗"
                cell = f"{tick} {r_match.predicted_family} ({r_match.confidence:.0%})"
                print(f"  {cell:<14}", end="")
        print()

    # ── Strategy summary ───────────────────────────────────────────────────────
    print(f"\n  {'Strategy':<16}  {'Accuracy':>9}  {'Avg queries':>12}  "
          f"{'Suite %':>8}  {'Avg time':>10}")
    print(f"  {'─'*16}  {'─'*9}  {'─'*12}  {'─'*8}  {'─'*10}")
    for s in summaries:
        print(f"  {s.strategy:<16}  {s.accuracy:>8.1%}  "
              f"{s.avg_queries:>12.1f}  {s.avg_query_pct:>7.1%}  "
              f"{s.avg_time:>9.1f}s")

    # ── Efficiency insight ─────────────────────────────────────────────────────
    full = next((s for s in summaries if s.strategy == "full-suite"), None)
    if full:
        print(f"\n  Early-stop efficiency vs full-suite:")
        for s in summaries:
            if s.strategy == "full-suite" or s.avg_query_pct == 0:
                continue
            saved  = 1.0 - s.avg_query_pct
            acc_delta = s.accuracy - full.accuracy
            sign = "+" if acc_delta >= 0 else ""
            print(f"    {s.strategy:<16}  saves {saved:.0%} of queries  "
                  f"accuracy delta: {sign}{acc_delta:.1%}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM fingerprinting strategies on known models."
    )
    parser.add_argument(
        "--models-file", required=True,
        help=(
            "JSON file listing test models. "
            'Format: [{"name": ..., "family": ..., "endpoint": ..., "backend": ...}, ...]'
        ),
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES,
        help=f"Strategies to benchmark (default: all). Choices: {STRATEGIES}"
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Prompt repeats per fingerprinting run (default: 1)"
    )
    parser.add_argument(
        "--classifier-path",
        default=str(config.MODEL_DIR / "classifier_model.joblib"),
        help="Path to trained ensemble classifier"
    )
    parser.add_argument(
        "--templates-path",
        default=str(config.TEMPLATES_PATH),
        help="Path to family templates joblib"
    )
    parser.add_argument(
        "--output", help="Save JSON results to this file"
    )
    args = parser.parse_args()

    # ── Load model list ────────────────────────────────────────────────────────
    try:
        with open(args.models_file) as f:
            raw = json.load(f)
    except Exception as e:
        print(f"\nERROR loading {args.models_file}: {e}\n")
        sys.exit(1)

    model_specs = [ModelSpec(**m) for m in raw]
    print(f"\nLoaded {len(model_specs)} test model(s) from {args.models_file}")
    print(f"Strategies: {', '.join(args.strategies)}\n")

    # ── Validate API keys before touching any endpoint ─────────────────────────
    KEY_ENV: dict[str, str] = {
        "openai":       "OPENAI_API_KEY",
        "ollama-cloud": "OLLAMA_CLOUD_API_KEY",
        "deepseek":     "DEEPSEEK_API_KEY",
        "gemini":       "GEMINI_API_KEY",
    }
    missing = []
    for spec in model_specs:
        env_var = KEY_ENV.get(spec.backend)
        if env_var and not spec.api_key and not os.environ.get(env_var):
            missing.append(f"  {spec.name} ({spec.backend}) — set {env_var} or add \"api_key\" to {args.models_file}")
    if missing:
        print("ERROR: Missing API keys for the following models:")
        for m in missing:
            print(m)
        print()
        sys.exit(1)

    # Show which key source is being used (masked)
    for spec in model_specs:
        env_var = KEY_ENV.get(spec.backend)
        if env_var:
            key_val = spec.api_key or os.environ.get(env_var, "")
            source  = "models file" if spec.api_key else f"${env_var}"
            masked  = key_val[:8] + "..." if len(key_val) > 8 else "(empty!)"
            print(f"  {spec.name}: using key from {source} ({masked})")

    # ── Load classifier ────────────────────────────────────────────────────────
    classifier = EnsembleClassifier()
    if not classifier.load(args.classifier_path):
        print(f"ERROR: Could not load classifier from {args.classifier_path}")
        print("Run 'llm-fingerprinter train' first.\n")
        sys.exit(1)

    # ── Load templates (optional — only needed for template-only strategy) ─────
    tc = TemplateClassifier()
    if "template-only" in args.strategies:
        if not tc.load(args.templates_path):
            print(f"WARNING: Could not load templates from {args.templates_path}")
            print("Skipping 'template-only' strategy.\n")
            args.strategies = [s for s in args.strategies if s != "template-only"]

    # ── Build per-endpoint fingerprinters (reuse across strategies) ────────────
    fingerprinters: dict[str, LLMFingerprinter] = {}
    suite     = PromptSuite()
    extractor = FeatureExtractor()

    for spec in model_specs:
        key = f"{spec.endpoint}::{spec.backend}"
        if key not in fingerprinters:
            client = build_client(spec.endpoint, spec.backend, spec.api_key)
            fingerprinters[key] = LLMFingerprinter(
                endpoint=spec.endpoint,
                ollama_client=client,
                prompt_suite=suite,
                feature_extractor=extractor,
                classifier=classifier,
            )

    # ── Run benchmark ──────────────────────────────────────────────────────────
    all_summaries: list[BenchmarkSummary] = []

    for strategy in args.strategies:
        print(f"Running strategy: {strategy} ...")
        strategy_results: list[StrategyResult] = []

        early_stop: Optional[float] = None
        if strategy == "early-stop-90":
            early_stop = 0.90
        elif strategy == "early-stop-85":
            early_stop = 0.85

        for spec in model_specs:
            key = f"{spec.endpoint}::{spec.backend}"
            fp  = fingerprinters[key]
            print(f"  {spec.name} ({spec.family}) ...", end=" ", flush=True)

            if strategy == "template-only":
                r = run_template_strategy(spec, fp, tc, args.repeats)
            else:
                r = run_ensemble_strategy(spec, fp, classifier,
                                          early_stop, args.repeats)

            status = "✓" if r.correct else ("ERR" if r.error else "✗")
            print(f"{status}  {r.queries_used}q  {r.elapsed_seconds:.1f}s")
            strategy_results.append(r)

        all_summaries.append(summarize(strategy, strategy_results))

    # ── Print report ───────────────────────────────────────────────────────────
    print_report(all_summaries)

    # ── Optionally save JSON ───────────────────────────────────────────────────
    if args.output:
        out_data = []
        for s in all_summaries:
            entry = {
                "strategy": s.strategy,
                "n_models": s.n_models,
                "accuracy": s.accuracy,
                "avg_queries": s.avg_queries,
                "avg_query_pct": s.avg_query_pct,
                "avg_time": s.avg_time,
                "results": [
                    {
                        "model": r.model_name,
                        "true_family": r.true_family,
                        "predicted_family": r.predicted_family,
                        "correct": r.correct,
                        "confidence": r.confidence,
                        "queries_used": r.queries_used,
                        "queries_total": r.queries_total,
                        "elapsed_seconds": r.elapsed_seconds,
                        "error": r.error,
                    }
                    for r in s.results
                ],
            }
            out_data.append(entry)

        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
