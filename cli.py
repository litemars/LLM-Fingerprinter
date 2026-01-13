#!/usr/bin/env python3
"""
Common workflows/commands
1. simulate  - Fingerprint known models with --family labels
2. train     - Build classifier from fingerprints  
3. identify  - Classify unknown models
"""

import click
import logging
import sys
import os
from pathlib import Path
from datetime import datetime


# Setup path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_fingerprint.ollama_client import OllamaClient
from llm_fingerprint.openai_client import OpenAIClient, OpenAIAuthError
from llm_fingerprint.ollama_cloud_client import OllamaCloudClient, OllamaCloudAuthError
from llm_fingerprint.deepseek_client import DeepSeekClient, DeepSeekAuthError
from llm_fingerprint.gemini_client import GeminiClient, GeminiAuthError
from llm_fingerprint.template_client import TemplateClient, TemplateAuthError
from llm_fingerprint.prompt_suite import PromptSuite
from llm_fingerprint.feature_extractor import FeatureExtractor
from llm_fingerprint.classifier import EnsembleClassifier, create_classifier
from llm_fingerprint.fingerprinter import LLMFingerprinter
from llm_fingerprint.fingerprint_store import FingerprintStore
from llm_fingerprint import config


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else getattr(logging, config.LOG_LEVEL, logging.INFO)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / f"fingerprinter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=level,
        format=config.LOG_FORMAT,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
    )

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def get_default_endpoint(backend):
    return {
        "ollama": config.OLLAMA_DEFAULT_ENDPOINT,
        "ollama-cloud": config.OLLAMA_CLOUD_DEFAULT_ENDPOINT,
        "openai": config.OPENAI_DEFAULT_ENDPOINT,
        "deepseek": config.DEEPSEEK_DEFAULT_ENDPOINT,
        "gemini": config.GEMINI_DEFAULT_ENDPOINT,
        "custom": "http://localhost:8000/v1",
    }.get(backend, config.OLLAMA_DEFAULT_ENDPOINT)


def get_api_client(backend, endpoint, api_key = None):
    if not api_key and backend in config.API_KEY_ENV_VARS:
        api_key = os.environ.get(config.API_KEY_ENV_VARS[backend])
    
    if backend == "ollama":
        return OllamaClient(endpoint=endpoint)
    elif backend == "ollama-cloud":
        if not api_key:
            raise click.ClickException("Ollama Cloud API key required")
        return OllamaCloudClient(api_key=api_key, endpoint=endpoint)
    
    elif backend == "openai":
        if not api_key:
            raise click.ClickException("OpenAI API key required")
        return OpenAIClient(api_key=api_key, endpoint=endpoint)
    
    elif backend == "deepseek":
        if not api_key:
            raise click.ClickException(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY or use --api-key"
            )
        return DeepSeekClient(api_key=api_key, endpoint=endpoint)
    
    elif backend == "gemini":
        if not api_key:
            raise click.ClickException(
                "Gemini API key required. Set GEMINI_API_KEY or use --api-key"
            )
        return GeminiClient(api_key=api_key, endpoint=endpoint)
    elif backend == "custom":
        if not api_key:
            raise click.ClickException("Custom API key required")
        return TemplateClient(api_key=api_key, endpoint=endpoint)
    
    raise click.ClickException(f"Unknown backend: {backend}")


def print_header():
    """Print application header."""
    click.echo("""
╔════════════════════════════════════════════════════════════════╗
║                 LLM FINGERPRINTING SYSTEM                      ║
║               Black-box Model Identification                   ║
╚════════════════════════════════════════════════════════════════╝
""")


def print_report(result: dict):
    click.echo("\n" + "=" * 60)
    click.echo("                 IDENTIFICATION REPORT")
    click.echo("=" * 60)

    if 'error' in result:
        click.echo(click.style(f"\n❌ Error: {result['error']}", fg='red'))
        return

    family = result.get('family', 'Unknown')
    confidence = result.get('confidence', 0.0)
    
    conf_color = 'green' if confidence > 0.7 else 'yellow' if confidence > 0.4 else 'red'
    click.echo(f"\n  Identified: {click.style(family.upper(), fg=conf_color, bold=True)}")
    click.echo(f"  Confidence: {click.style(f'{confidence*100:.1f}%', fg=conf_color)}")

    all_probs = result.get('all_probabilities', {})
    if all_probs:
        click.echo("\n  Probabilities:")
        for fam, prob in sorted(all_probs.items(), key=lambda x: -x[1])[:5]:
            bar = "█" * int(prob * 25)
            click.echo(f"    {fam:12s} {prob*100:5.1f}% {bar}")
    
    click.echo("=" * 60)


def backend_options(f):
    f = click.option('--backend', '-b', type=click.Choice(['ollama', 'ollama-cloud', 'openai', 'custom', 'gemini', 'deepseek']), 
                     default='ollama', help='API backend')(f)
    f = click.option('--endpoint', '-e', default=None, help='API endpoint URL')(f)
    f = click.option('--api-key', '-k', default=None, help='API key')(f)
    return f


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, verbose):
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['logger'] = setup_logging(verbose)


@cli.command()
@backend_options
@click.option('--model', required=True, help='Model name')
@click.option('--repeats', default=1, type=int, help='Prompt repeats (default: 1)')
@click.pass_context
def identify(ctx, backend, endpoint, api_key, model, repeats):
    print_header()
    logger = ctx.obj['logger']
    endpoint = endpoint or get_default_endpoint(backend)

    try:
        client = get_api_client(backend, endpoint, api_key)
        if not client._check_connectivity():
            click.echo(click.style(f"❌ Cannot reach {backend} at {endpoint}", fg='red'))
            sys.exit(1)

        click.echo(f"🔧 Initializing ({backend})...")
        suite = PromptSuite()
        extractor = FeatureExtractor()
        classifier = EnsembleClassifier(config.MODEL_FAMILIES)
        fingerprinter = LLMFingerprinter(endpoint, client, suite, extractor, classifier)

        classifier_path = config.MODEL_DIR / "classifier_model.joblib"
        if classifier_path.exists():
            classifier.load(str(classifier_path))
            mode = "PCA" if classifier.use_pca else "raw features"
            dims = classifier.input_dim or "?"
            click.echo(f"📂 Loaded classifier ({mode}, {dims} dims)")
        else:
            click.echo(click.style("❌ No trained classifier found", fg='red'))
            click.echo("   Run 'simulate' then 'train' first")
            sys.exit(1)

        click.echo(f"\n📊 Fingerprinting {model}...")
        result = fingerprinter.identify(model, repeats=repeats)
        print_report(result)
        click.echo("\n✅ Done!")

    except (OpenAIAuthError, OllamaCloudAuthError, TemplateAuthError) as e:
        click.echo(click.style(f"❌ Auth failed: {e}", fg='red'))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        logger.exception("Failed")
        sys.exit(1)


@cli.command()
@backend_options
@click.option('--model', required=True, help='Model name')
@click.option('--family', required=True, type=click.Choice(list(config.MODEL_FAMILIES.keys())), help='Model family')
@click.option('--num-sims', default=3, type=int, help='Number of simulations')
@click.option('--repeats', default=2, type=int, help='Prompt repeats per simulation')
@click.pass_context
def simulate(ctx, backend, endpoint, api_key, model, family, num_sims, repeats):
    print_header()
    logger = ctx.obj['logger']
    endpoint = endpoint or get_default_endpoint(backend)

    try:
        client = get_api_client(backend, endpoint, api_key)
        if not client._check_connectivity():
            click.echo(click.style(f"❌ Cannot reach {backend}", fg='red'))
            sys.exit(1)

        click.echo(f"🔧 Initializing ({backend})...")
        suite = PromptSuite()
        extractor = FeatureExtractor()
        classifier = EnsembleClassifier(config.MODEL_FAMILIES)
        fingerprinter = LLMFingerprinter(endpoint, client, suite, extractor, classifier)
        store = FingerprintStore(str(config.FINGERPRINTS_DIR))

        click.echo(f"\n🔄 Running {num_sims} simulations for {model} ({family})...")
        
        success_count = 0
        for sim_idx in range(num_sims):
            click.echo(f"\n  Simulation {sim_idx + 1}/{num_sims}:")
            fp = fingerprinter.fingerprint_model(model, repeats=repeats)
            
            if fp is None:
                click.echo(click.style(f"    ⚠️ Failed", fg='yellow'))
                continue

            fp['metadata']['family'] = family
            fp['metadata']['backend'] = backend
            fp_path = store.save_fingerprint(fp, f"{family}_sim_{sim_idx}", family=family)
            click.echo(f"    ✅ Saved: {fp_path.name} ({len(fp['vector'])} dims)")
            success_count += 1

        click.echo(f"\n✅ Completed {success_count}/{num_sims} simulations")
        click.echo("   Next: Run 'train' to build classifier")

    except (OpenAIAuthError, OllamaCloudAuthError, TemplateAuthError) as e:
        click.echo(click.style(f"❌ Auth failed: {e}", fg='red'))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        logger.exception("Failed")
        sys.exit(1)


@cli.command()
@click.option('--augment/--no-augment', default=True, help='Data augmentation')
@click.option('--use-pca', is_flag=True, default=False, 
              help='Use PCA reduction (default: use raw 402-dim features)')
@click.option('--pca-components', default=64, type=int, 
              help='PCA components if --use-pca (default: 64)')
@click.pass_context
def train(ctx, augment, use_pca, pca_components):
    """
    Examples/Option:
      train                    # Raw features (recommended)
      train --use-pca          # With PCA (64 components)
      train --use-pca --pca-components 32
    """
    print_header()
    logger = ctx.obj['logger']

    mode = f"PCA ({pca_components} components)" if use_pca else "raw features (402-dim)"
    click.echo(f"🔧 Training mode: {mode}")

    try:
        store = FingerprintStore(str(config.FINGERPRINTS_DIR))
        click.echo("\n📂 Loading fingerprints...")
        training_data = store.export_for_training()

        if not training_data:
            click.echo(click.style("❌ No fingerprints found", fg='red'))
            click.echo("   Run 'simulate' first")
            sys.exit(1)

        click.echo("\n📊 Training data:")
        total = 0
        for fam, vecs in sorted(training_data.items()):
            dims = len(vecs[0]) if vecs else 0
            click.echo(f"    {fam}: {len(vecs)} samples ({dims} dims)")
            total += len(vecs)
        click.echo(f"    Total: {total}")

        if total < 2:
            click.echo(click.style("❌ Need at least 2 samples", fg='red'))
            sys.exit(1)

        click.echo(f"\n🧠 Training classifier ({mode})...")
        clf = create_classifier(
            model_families=config.MODEL_FAMILIES,
            use_pca=use_pca,
            pca_components=pca_components,
            augment_data=augment,
            augment_samples=config.AUGMENTATION_SAMPLES_PER_ORIGINAL if augment else 0
        )

        if not clf.train_from_simulations(training_data):
            click.echo(click.style("❌ Training failed", fg='red'))
            sys.exit(1)

        classifier_path = config.MODEL_DIR / "classifier_model.joblib"
        clf.save(str(classifier_path))
        
        click.echo(f"\n✅ Classifier trained and saved!")
        click.echo(f"   Mode: {mode}")
        click.echo(f"   Input dim: {clf.input_dim}")
        click.echo("   Run: identify --model <model-name>")

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        logger.exception("Failed")
        sys.exit(1)


@cli.command('list-models')
@backend_options
@click.pass_context
def list_models(ctx, backend, endpoint, api_key):
    """List available models."""
    print_header()
    endpoint = endpoint or get_default_endpoint(backend)

    try:
        client = get_api_client(backend, endpoint, api_key)
        if not client._check_connectivity():
            click.echo(click.style(f"❌ Cannot reach {backend}", fg='red'))
            sys.exit(1)

        models = client.list_models()
        if not models:
            click.echo("No models found")
            return

        click.echo(f"📦 Models on {backend}:\n")
        for i, m in enumerate(models, 1):
            click.echo(f"  {i:2}. {m}")
        click.echo(f"\nTotal: {len(models)}")

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        sys.exit(1)


@cli.command('list-fingerprints')
def list_fingerprints():
    """List saved fingerprints."""
    print_header()
    store = FingerprintStore(str(config.FINGERPRINTS_DIR))
    counts = store.count_by_family()

    if not counts:
        click.echo("No fingerprints found")
        return

    click.echo("📚 Fingerprints:\n")
    total = 0
    for fam in sorted(counts.keys()):
        cnt = counts[fam]
        total += cnt
        click.echo(f"  {fam:12s} {cnt:3d} {'█' * min(cnt, 20)}")
    click.echo(f"\n  Total: {total}")
    
    # Check classifier status
    classifier_path = config.MODEL_DIR / "classifier_model.joblib"
    if classifier_path.exists():
        try:
            import joblib
            data = joblib.load(classifier_path)
            mode = "PCA" if data.get('use_pca', False) else "raw features"
            dims = data.get('input_dim', '?')
            click.echo(f"\n✅ Classifier trained ({mode}, {dims} dims)")
        except:
            click.echo("\n✅ Classifier available")
    else:
        click.echo("\n⚠️  No classifier. Run 'train'")


@cli.command()
def info():
    """Show system info."""
    print_header()
    
    click.echo("⚙️  Config:")
    click.echo(f"  Fingerprints: {config.FINGERPRINTS_DIR}")
    click.echo(f"  Embedding:    {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIM}d)")
    click.echo(f"  Total dims:   {config.TOTAL_FEATURE_DIM} (384 + 12 + 6)")
    
    click.echo(f"\n🔌 Backends:")
    click.echo(f"  Ollama:       {config.OLLAMA_DEFAULT_ENDPOINT}")
    click.echo(f"  Ollama Cloud: {config.OLLAMA_CLOUD_DEFAULT_ENDPOINT}")
    click.echo(f"  OpenAI:       {config.OPENAI_DEFAULT_ENDPOINT}")
    click.echo(f"  deepseek:     {config.DEEPSEEK_DEFAULT_ENDPOINT}")
    click.echo(f"  gemini:       {config.GEMINI_DEFAULT_ENDPOINT}")
    
    click.echo(f"\n📋 Families: {', '.join(sorted(config.MODEL_FAMILIES.keys()))}")
    
    store = FingerprintStore(str(config.FINGERPRINTS_DIR))
    counts = store.count_by_family()
    classifier_path = config.MODEL_DIR / "classifier_model.joblib"
    
    click.echo(f"\n📊 Status:")
    click.echo(f"  Fingerprints: {sum(counts.values())}")
    
    if classifier_path.exists():
        try:
            import joblib
            data = joblib.load(classifier_path)
            mode = "PCA" if data.get('use_pca', False) else "raw features"
            dims = data.get('input_dim', '?')
            click.echo(f"  Classifier:   ✅ trained ({mode}, {dims} dims)")
        except:
            click.echo(f"  Classifier:   ✅ trained")
    else:
        click.echo(f"  Classifier:   ❌ not trained")
    
    click.echo(f"\n💡 Training options:")
    click.echo(f"  train              # Use raw 402-dim features (default)")
    click.echo(f"  train --use-pca    # Use PCA reduction (64 dims)")


@cli.command()
@backend_options
@click.option('--model', required=True, help='Model name')
@click.option('--repeats', default=1, type=int, help='Prompt repeats (default: 1)')
@click.option('--output', type=click.Path(), help='Output directory')
@click.pass_context
def fingerprint(ctx, backend, endpoint, api_key, model, repeats, output):
    """Generate fingerprint without classification."""
    print_header()
    logger = ctx.obj['logger']
    endpoint = endpoint or get_default_endpoint(backend)

    try:
        client = get_api_client(backend, endpoint, api_key)
        if not client._check_connectivity():
            click.echo(click.style(f"❌ Cannot reach {backend}", fg='red'))
            sys.exit(1)

        click.echo(f"🔧 Initializing ({backend})...")
        suite = PromptSuite()
        extractor = FeatureExtractor()
        classifier = EnsembleClassifier(config.MODEL_FAMILIES)
        fingerprinter = LLMFingerprinter(endpoint, client, suite, extractor, classifier)

        click.echo(f"\n📊 Fingerprinting {model}...")
        fp = fingerprinter.fingerprint_model(model, repeats=repeats)

        if fp is None:
            click.echo(click.style("❌ Failed", fg='red'))
            sys.exit(1)

        click.echo(f"\n✅ Generated:")
        click.echo(f"   Dimension: {len(fp['vector'])}")
        click.echo(f"   Queries:   {fp['metadata']['queries_executed']}")
        click.echo(f"   Duration:  {fp['metadata']['duration_seconds']:.1f}s")

        if output:
            store = FingerprintStore(output)
            path = store.save_fingerprint(fp, model)
            click.echo(f"\n📁 Saved: {path}")

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        logger.exception("Failed")
        sys.exit(1)


if __name__ == '__main__':
    cli()
