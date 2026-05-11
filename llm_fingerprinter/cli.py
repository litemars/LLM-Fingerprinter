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

import numpy as np

from llm_fingerprinter.ollama_client import OllamaClient
from llm_fingerprinter.openai_client import OpenAIClient, OpenAIAuthError
from llm_fingerprinter.ollama_cloud_client import OllamaCloudClient, OllamaCloudAuthError
from llm_fingerprinter.custom_client import CustomClient, CustomAuthError
from llm_fingerprinter.deepseek_client import DeepSeekClient, DeepSeekAuthError
from llm_fingerprinter.gemini_client import GeminiClient, GeminiAuthError
from llm_fingerprinter.prompt_suite import PromptSuite
from llm_fingerprinter.feature_extractor import FeatureExtractor
from llm_fingerprinter.classifier import EnsembleClassifier, create_classifier
from llm_fingerprinter.fingerprinter import LLMFingerprinter
from llm_fingerprinter.fingerprint_store import FingerprintStore
from llm_fingerprinter.template_classifier import TemplateClassifier
from llm_fingerprinter import config


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
        "ollama":       config.OLLAMA_DEFAULT_ENDPOINT,
        "ollama-cloud": config.OLLAMA_CLOUD_DEFAULT_ENDPOINT,
        "openai":       config.OPENAI_DEFAULT_ENDPOINT,
        "deepseek":     config.DEEPSEEK_DEFAULT_ENDPOINT,
        "gemini":       config.GEMINI_DEFAULT_ENDPOINT,
        "custom":       config.CUSTOM_DEFAULT_ENDPOINT,
    }.get(backend, config.CUSTOM_DEFAULT_ENDPOINT)


def get_api_client(backend, endpoint, api_key = None, request_file = None):
    if not api_key and backend in config.API_KEY_ENV_VARS:
        api_key = os.environ.get(config.API_KEY_ENV_VARS[backend])
    
    if backend == "ollama":
        return OllamaClient(endpoint=endpoint)
      
    elif backend == "ollama-cloud":
        if not api_key:
            raise click.ClickException("Ollama Cloud API key required. Set OLLAMA_CLOUD_API_KEY or use --api-key")
        return OllamaCloudClient(api_key=api_key, endpoint=endpoint)
    
    elif backend == "openai":
        if not api_key:
            raise click.ClickException("OpenAI API key required. Set OPENAI_API_KEY or use --api-key")
        return OpenAIClient(api_key=api_key, endpoint=endpoint)
    
    elif backend == "deepseek":
        if not api_key:
            raise click.ClickException("DeepSeek API key required. Set DEEPSEEK_API_KEY or use --api-key")
        return DeepSeekClient(api_key=api_key, endpoint=endpoint)
    
    elif backend == "gemini":
        if not api_key:
            raise click.ClickException("Gemini API key required. Set GEMINI_API_KEY or use --api-key")
        return GeminiClient(api_key=api_key, endpoint=endpoint)
    
    elif backend == "custom":
        if not request_file:
            raise click.ClickException("Custom backend requires --request-file (-r)")
        return CustomClient(request_file=request_file, api_key=api_key)
    
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

    family     = result.get('family', 'Unknown')
    confidence = result.get('confidence', 0.0)
    is_ood     = result.get('ood_detected', False)
    tc_result  = result.get('template_result')
    model_est  = result.get('model_estimate')

    # ── Family ────────────────────────────────────────────────────────────────
    family_source = result.get('family_source')
    if is_ood and family_source == 'model_template':
        # Family was recovered from a high-confidence model-template match —
        # show it prominently instead of the confusing OOD ensemble output.
        mt_conf = result.get('model_estimate', {}).get('confidence', 0.0)
        click.echo(f"\n  Family:     {click.style(family.upper(), fg='green', bold=True)}"
                   f"  ({mt_conf*100:.1f}% via model template)")
        click.echo(click.style(
            "  ⚠️  Note: ensemble was uncertain — family inferred from model match",
            fg='yellow'))
    elif is_ood:
        click.echo(click.style(f"\n  ⚠️  OUT-OF-DISTRIBUTION DETECTED", fg='yellow', bold=True))
        predicted = result.get('predicted_family', '?')
        click.echo(f"  Best guess: {click.style(predicted.upper(), fg='yellow')}")
        click.echo(f"  Confidence: {click.style(f'{confidence*100:.1f}%', fg='red')}")
        ood_details = result.get('ood_details', {})
        click.echo(f"  Classifier agreement: {ood_details.get('agreement_ratio', 0)*100:.0f}%")
        click.echo(click.style("  Model may not match any known family", fg='yellow'))
    else:
        conf_color = 'green' if confidence > 0.7 else 'yellow' if confidence > 0.4 else 'red'
        click.echo(f"\n  Family:     {click.style(family.upper(), fg=conf_color, bold=True)}"
                   f"  ({confidence*100:.1f}%)")

    # ── Ensemble probabilities ────────────────────────────────────────────────
    all_probs = result.get('all_probabilities', {})
    if all_probs:
        click.echo("\n  Probabilities:")
        for fam, prob in sorted(all_probs.items(), key=lambda x: -x[1])[:5]:
            bar = "█" * int(prob * 25)
            click.echo(f"    {fam:12s} {prob*100:5.1f}% {bar}")

    # ── Model estimate (specific version) ─────────────────────────────────────
    if model_est:
        best_model = model_est.get('predicted_model', '?')
        me_conf    = model_est.get('confidence', 0.0)
        me_ood     = model_est.get('is_ood', False)
        me_color   = 'green' if me_conf > 0.6 else 'yellow'
        # "ambiguous" means two candidate models scored very close — best guess
        # is shown but treat it with lower trust
        me_note    = click.style("  (ambiguous — two models too similar to distinguish)",
                                 fg='yellow') if me_ood else ""
        click.echo(f"\n  Model:      {click.style(best_model, fg=me_color, bold=True)}"
                   f"  ({me_conf*100:.1f}%){me_note}")
        click.echo("  Other candidates:")
        for rr in model_est.get('ranked', [])[1:4]:
            click.echo(f"    {rr['family']:26s} dist={rr['distance']:.4f}")

    # ── Template warning — ONLY shown when it adds new information ─────────────
    # (disagrees with ensemble, or flags the model as unknown/OOD)
    # In the normal case where template agrees, it stays silent.
    if tc_result:
        tc_predicted = tc_result.get('predicted_family', '').lower()
        tc_ood       = tc_result.get('is_ood', False)
        disagrees    = not is_ood and tc_predicted and tc_predicted != family.lower()
        if tc_ood:
            click.echo(click.style(
                "\n  ⚠️  Warning: model may not belong to any known family",
                fg='yellow'))
        elif disagrees:
            click.echo(click.style(
                f"\n  ⚠️  Warning: template classifier disagrees"
                f" — nearest family is {tc_predicted.upper()}",
                fg='yellow'))

    # Query usage / early stopping summary
    q_used = result.get('queries_executed', 0)
    q_total = result.get('queries_total', 0)
    early_stopped = result.get('early_stopped', False)
    layers_done = result.get('layers_completed', [])

    if q_total:
        click.echo(f"\n  Queries: {q_used}/{q_total} "
                   f"({'layers: ' + ', '.join(layers_done)})")
        if early_stopped:
            saved = q_total - q_used
            click.echo(click.style(
                f"  ⚡ Early stopped — saved {saved} queries ({saved/q_total*100:.0f}%)",
                fg='cyan'
            ))

    click.echo("=" * 60)


def backend_options(f):
    """Common backend options for all LLM commands."""
    f = click.option('--backend', '-b', 
                     type=click.Choice(['ollama', 'ollama-cloud', 'openai', 'custom', 'gemini', 'deepseek']), 
                     default=config.DEFAULT_BACKEND, help='API backend')(f)
    f = click.option('--endpoint', '-e', default=None, help='API endpoint URL')(f)
    f = click.option('--api-key', '-k', default=None, help='API key')(f)
    f = click.option('--request-file', '-r', default=None, type=click.Path(exists=True),
                     help='[custom] Request template file')(f)
    return f


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, verbose):
    """LLM Fingerprinting - Model Identification System.
    
    \b
    Workflow:
      1. simulate  - Fingerprint known models with --family labels
      2. train     - Build classifier from fingerprints  
      3. identify  - Classify unknown models
    
    \b
    Backends:
      ollama       - Local Ollama server (default)
      ollama-cloud - Ollama Cloud API
      openai       - OpenAI API
      deepseek     - DeepSeek API
      gemini       - Google Gemini API
      custom       - Any API via request template file
    
    \b
    Custom Backend Examples:
      llm-fingerprinter identify -b custom -r ./request.txt
      llm-fingerprinter identify -b custom -r ./request.txt -k my-api-key
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['logger'] = setup_logging(verbose)


@cli.command()
@backend_options
@click.option('--model', '-m', default=None, help='Model name')
@click.option('--repeats', default=1, type=int, help='Prompt repeats (default: 1)')
@click.option('--early-stop', default=0.0, type=float, show_default=True,
              metavar='THRESHOLD',
              help='Stop after a layer if confidence exceeds THRESHOLD. '
                   'Skips remaining layers and saves API calls. '
                   'Default 0 = disabled (always run full suite).')
@click.pass_context
def identify(ctx, backend, endpoint, api_key, request_file, model, repeats, early_stop):
    """Identify model family using trained classifier.

    \b
    Examples:
      identify -b ollama --model llama3.2
      identify -b ollama --model llama3.2 --early-stop 0.90
      identify -b openai --model gpt-4o-mini
      identify -b custom -r ./request.txt
      identify -b custom -r ./request.txt -k my-api-key
    """
    print_header()
    logger = ctx.obj['logger']
    
    # Validate options
    if backend == "custom":
        if not request_file:
            raise click.ClickException("Custom backend requires --request-file (-r)")
        click.echo(f"🔌 Backend: custom")
        click.echo(f"📄 Request file: {request_file}")
    else:
        endpoint = endpoint or get_default_endpoint(backend)
        click.echo(f"🔌 Backend: {backend}")
        click.echo(f"🌐 Endpoint: {endpoint}")

    try:
        client = get_api_client(backend, endpoint, api_key, request_file)
        if not client._check_connectivity():
            click.echo(click.style(f"❌ Cannot connect to API", fg='red'))
            sys.exit(1)
        click.echo("✅ Connected")

        click.echo(f"\n🔧 Initializing...")
        suite = PromptSuite()
        extractor = FeatureExtractor()
        classifier = EnsembleClassifier(config.MODEL_FAMILIES)
        fingerprinter = LLMFingerprinter(endpoint if backend != "custom" else "custom", 
                                         client, suite, extractor, classifier)

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

        model_display = model or "default"
        early_stop_threshold = early_stop if early_stop else None
        if early_stop_threshold:
            click.echo(f"\n📊 Fingerprinting {model_display} "
                       f"(early-stop @ {early_stop_threshold*100:.0f}% confidence)...")
        else:
            click.echo(f"\n📊 Fingerprinting {model_display} (full suite)...")
        result = fingerprinter.identify(model, repeats=repeats,
                                        early_stop_confidence=early_stop_threshold)

        fp_vec = result.get('fingerprint', {}).get('vector')

        # Model-level template — specific model identification
        if config.MODEL_TEMPLATES_PATH.exists() and fp_vec is not None:
            mtc = TemplateClassifier()
            if mtc.load(str(config.MODEL_TEMPLATES_PATH)):
                try:
                    mt_res = mtc.classify(fp_vec, top_k=4)
                    result['model_estimate'] = {
                        'predicted_model':  mt_res['predicted_family'],
                        'confidence':       mt_res['confidence'],
                        'distance':         mt_res['distance'],
                        'is_ood':           mt_res['is_ood'],
                        'ranked':           mt_res['ranked'],
                        'inferred_family':  mt_res.get('inferred_family'),
                    }
                    click.echo(f"🔬 Model templates: {len(mtc.templates)} models loaded")

                    # ── Family recovery from model template ───────────────
                    # When the ensemble is OOD (uncertain) but the model-
                    # template match is clear and high-confidence, trust the
                    # family label that was stored at build-model-templates
                    # time rather than the ensemble's confused best guess.
                    me = result['model_estimate']
                    if (result.get('ood_detected')
                            and not me['is_ood']
                            and me['confidence'] >= 0.8
                            and me.get('inferred_family')):
                        result['family'] = me['inferred_family']
                        result['family_source'] = 'model_template'
                        logger.info(
                            f"Family recovered from model template: "
                            f"{me['inferred_family']} "
                            f"(confidence={me['confidence']:.3f})"
                        )
                except Exception as _mt_err:
                    logger.debug(f"Model template classify failed: {_mt_err}")

        # Family-level template classifier — optional open-set second opinion
        if config.TEMPLATES_PATH.exists() and fp_vec is not None:
            tc = TemplateClassifier()
            if tc.load(str(config.TEMPLATES_PATH)):
                try:
                    result['template_result'] = tc.classify(fp_vec)
                    click.echo(f"📐 Family templates: {len(tc.templates)} families")
                except Exception as _tc_err:
                    logger.debug(f"Template classify failed: {_tc_err}")

        print_report(result)
        click.echo("\n✅ Done!")

    except (OpenAIAuthError, OllamaCloudAuthError, DeepSeekAuthError, GeminiAuthError, CustomAuthError) as e:
        click.echo(click.style(f"❌ Auth failed: {e}", fg='red'))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        logger.exception("Failed")
        sys.exit(1)


@cli.command()
@backend_options
@click.option('--model', '-m', default=None, help='Model name')
@click.option('--family', '-f', required=True, type=click.Choice(list(config.MODEL_FAMILIES.keys())), help='Model family')
@click.option('--num-sims', '-n', default=3, type=int, help='Number of simulations')
@click.option('--repeats', default=1, type=int, help='Prompt repeats per simulation (default: 1)')
@click.pass_context
def simulate(ctx, backend, endpoint, api_key, request_file, model, family, num_sims, repeats):
    """Run fingerprinting simulations for training.
    
    \b
    Examples:
      simulate -b ollama --model llama3.2 --family llama
      simulate -b openai --model gpt-4o-mini --family gpt
      simulate -b custom -r ./request.txt --family gpt
      simulate -b custom -r ./request.txt -k my-api-key --family llama
    """
    print_header()
    logger = ctx.obj['logger']
    
    # Validate options
    if backend == "custom":
        if not request_file:
            raise click.ClickException("Custom backend requires --request-file (-r)")
        click.echo(f"🔌 Backend: custom")
        click.echo(f"📄 Request file: {request_file}")
    else:
        endpoint = endpoint or get_default_endpoint(backend)
        click.echo(f"🔌 Backend: {backend}")
        click.echo(f"🌐 Endpoint: {endpoint}")

    try:
        client = get_api_client(backend, endpoint, api_key, request_file)
        if not client._check_connectivity():
            click.echo(click.style(f"❌ Cannot connect to API", fg='red'))
            sys.exit(1)
        click.echo("✅ Connected")

        click.echo(f"\n🔧 Initializing...")
        suite = PromptSuite()
        extractor = FeatureExtractor()
        classifier = EnsembleClassifier(config.MODEL_FAMILIES)
        fingerprinter = LLMFingerprinter(endpoint if backend != "custom" else "custom",
                                         client, suite, extractor, classifier)
        store = FingerprintStore(str(config.TRAINING_DIR))

        model_display = model or "default"
        click.echo(f"\n🔄 Running {num_sims} simulations for {model_display} ({family})...")

        temperatures = np.linspace(0.0, 1.0, num_sims).tolist() if num_sims > 1 else [config.TEMPERATURE]
        click.echo(f"   Temperatures: {[round(t, 2) for t in temperatures]}")

        success_count = 0
        for sim_idx in range(num_sims):
            temp = round(temperatures[sim_idx], 2)
            click.echo(f"\n  Simulation {sim_idx + 1}/{num_sims} (temp={temp}):")
            fp = fingerprinter.fingerprint_model(model, repeats=repeats, temperature=temp)

            if fp is None:
                click.echo(click.style(f"    ⚠️ Failed (all prompts returned errors)", fg='yellow'))
                continue

            fp['metadata']['model_name'] = model_display
            fp['metadata']['family'] = family
            fp['metadata']['backend'] = backend
            save_name = f"{model_display}_{family}_sim{sim_idx}_t{int(temp*100)}"
            fp_path = store.save_fingerprint(fp, save_name, family=family)
            click.echo(f"    ✅ Saved: {fp_path.name} ({len(fp['vector'])} dims)")
            success_count += 1

        if success_count == 0:
            click.echo(click.style(f"\n❌ All {num_sims} simulations failed", fg='red'))
            sys.exit(1)

        click.echo(f"\n✅ Completed {success_count}/{num_sims} simulations")
        click.echo("   Next: Run 'train' to build classifier")

    except (OpenAIAuthError, OllamaCloudAuthError, DeepSeekAuthError, GeminiAuthError, CustomAuthError) as e:
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
@click.option('--cross-validate', '-cv', is_flag=True, default=False,
              help='Run k-fold cross-validation before training')
@click.option('--cv-folds', default=5, type=int,
              help='Number of cross-validation folds (default: 5)')
@click.pass_context
def train(ctx, augment, use_pca, pca_components, cross_validate, cv_folds):
    """Train classifier from saved fingerprints.

    \b
    Examples:
      train                        # Raw features (recommended)
      train --use-pca              # With PCA (64 components)
      train --cross-validate       # With 5-fold cross-validation
      train -cv --cv-folds 3       # With 3-fold cross-validation
    """
    print_header()
    logger = ctx.obj['logger']

    mode = f"PCA ({pca_components} components)" if use_pca else "rebalanced features (per-layer)"
    click.echo(f"🔧 Training mode: {mode}")

    try:
        # Load from new training dir, fall back to legacy fingerprints dir
        training_data = {}
        for search_dir in [config.TRAINING_DIR, config.FINGERPRINTS_DIR]:
            store = FingerprintStore(str(search_dir))
            click.echo(f"\n📂 Loading fingerprints from {search_dir.name}/...")
            data = store.export_for_training()
            for family, vectors in data.items():
                training_data.setdefault(family, []).extend(vectors)

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

        # Cross-validation
        if cross_validate:
            click.echo(f"\n📈 Running {cv_folds}-fold cross-validation...")
            import numpy as np
            X_list, y_list = [], []
            for family_name, vectors in training_data.items():
                if family_name not in config.MODEL_FAMILIES:
                    continue
                class_id = config.MODEL_FAMILIES[family_name]
                for v in vectors:
                    X_list.append(np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v)
                    y_list.append(class_id)

            X_cv = np.array(X_list, dtype=np.float32)
            y_cv = np.array(y_list)

            cv_results = clf.cross_validate(X_cv, y_cv, n_folds=cv_folds)
            if cv_results:
                click.echo(f"\n   Mean accuracy: {cv_results['mean_accuracy']:.1%} "
                          f"({cv_results['n_folds']} folds)")
                click.echo(f"\n   Per-family metrics:")
                click.echo(f"   {'Family':12s} {'Prec':>6s} {'Recall':>8s} {'F1':>6s} {'Support':>8s}")
                click.echo(f"   {'-'*42}")
                for fam, metrics in sorted(cv_results['per_family'].items()):
                    click.echo(f"   {fam:12s} {metrics['precision']:6.3f} {metrics['recall']:8.3f} "
                              f"{metrics['f1']:6.3f} {metrics['support']:8d}")

                click.echo(f"\n   Fold accuracies: "
                          + ", ".join(f"{a:.1%}" for a in cv_results['fold_accuracies']))
            else:
                click.echo(click.style("   ⚠️ Not enough samples per class for cross-validation", fg='yellow'))

        classifier_path = config.MODEL_DIR / "classifier_model.joblib"
        clf.save(str(classifier_path))

        click.echo(f"\n✅ Classifier trained and saved!")
        click.echo(f"   Mode: {mode}")
        click.echo(f"   Input dim: {clf.input_dim}")
        click.echo("\n💡 Next steps:")
        click.echo("   build-templates        # enable open-set family detection + add-family")
        click.echo("   build-model-templates  # enable specific model version identification")
        click.echo("   identify --model <model-name>")

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        logger.exception("Failed")
        sys.exit(1)


@cli.command('list-models')
@backend_options
@click.pass_context
def list_models(ctx, backend, endpoint, api_key, request_file):
    """List available models on backend."""
    print_header()
    
    if backend == "custom":
        click.echo("⚠️  list-models not supported for custom backend")
        return
    
    endpoint = endpoint or get_default_endpoint(backend)

    try:
        client = get_api_client(backend, endpoint, api_key, request_file)
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

    # Aggregate counts from training dir and legacy dir
    family_counts = {}
    model_counts  = {}
    for directory in [config.TRAINING_DIR, config.FINGERPRINTS_DIR]:
        store = FingerprintStore(str(directory))
        for fam, cnt in store.count_by_family().items():
            family_counts[fam] = family_counts.get(fam, 0) + cnt
        for mdl, vecs in store.export_by_model().items():
            model_counts[mdl] = model_counts.get(mdl, 0) + len(vecs)

    if not family_counts:
        click.echo("No fingerprints found")
        return

    # ── By family ────────────────────────────────────────────────────────────
    click.echo("📚 By family:\n")
    total = 0
    for fam in sorted(family_counts.keys()):
        cnt = family_counts[fam]
        total += cnt
        click.echo(f"  {fam:12s} {cnt:3d}  {'█' * min(cnt, 20)}")
    click.echo(f"\n  Total: {total}")

    # ── By model (for build-model-templates) ─────────────────────────────────
    known_models = {m: c for m, c in model_counts.items()
                    if not any(x in m for x in ('_sim', '_t0', '_t25', '_t50', '_t75', '_t100'))}
    if known_models:
        click.echo("\n🔬 By model (for build-model-templates):\n")
        for mdl in sorted(known_models.keys()):
            cnt = known_models[mdl]
            flag = "" if cnt >= 3 else click.style("  ⚠️ <3 samples", fg='yellow')
            click.echo(f"  {mdl:28s} {cnt:3d}{flag}")

    # ── Classifier / template status ──────────────────────────────────────────
    click.echo("")
    classifier_path = config.MODEL_DIR / "classifier_model.joblib"
    if classifier_path.exists():
        try:
            import joblib
            data = joblib.load(classifier_path)
            mode = "PCA" if data.get('use_pca', False) else "raw features"
            dims = data.get('input_dim', '?')
            click.echo(f"✅ Classifier trained ({mode}, {dims} dims)")
        except Exception:
            click.echo("✅ Classifier available")
    else:
        click.echo("⚠️  No classifier — run 'train'")

    if config.TEMPLATES_PATH.exists():
        tc = TemplateClassifier()
        tc.load(str(config.TEMPLATES_PATH))
        click.echo(f"✅ Family templates built ({len(tc.templates)} families: "
                   f"{', '.join(sorted(tc.templates))})")
    else:
        click.echo("⚠️  No family templates — run 'build-templates'")

    if config.MODEL_TEMPLATES_PATH.exists():
        mtc = TemplateClassifier()
        mtc.load(str(config.MODEL_TEMPLATES_PATH))
        click.echo(f"✅ Model templates built  ({len(mtc.templates)} models: "
                   f"{', '.join(sorted(mtc.templates))})")
    else:
        click.echo("⚠️  No model templates  — run 'build-model-templates'")


@cli.command()
def info():
    """Show system info."""
    print_header()
    
    click.echo("⚙️  Config:")
    click.echo(f"  Fingerprints: {config.FINGERPRINTS_DIR}")
    click.echo(f"  Embedding:    {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIM}d)")
    click.echo(f"  Per-prompt:   {config.PER_PROMPT_FEATURE_DIM}d (384 + 12 + 6)")
    click.echo(f"  Fingerprint:  {config.RAW_FINGERPRINT_DIM}d ({config.NUM_PROMPT_LAYERS} layers x {config.PER_PROMPT_FEATURE_DIM})")
    click.echo(f"  Rebalanced:   {config.EMBEDDING_PCA_DIM}d embeddings per layer")

    click.echo(f"\n🔌 Backends:")
    click.echo(f"  ollama:       {config.OLLAMA_DEFAULT_ENDPOINT}")
    click.echo(f"  ollama-cloud: {config.OLLAMA_CLOUD_DEFAULT_ENDPOINT}")
    click.echo(f"  openai:       {config.OPENAI_DEFAULT_ENDPOINT}")
    click.echo(f"  deepseek:     {config.DEEPSEEK_DEFAULT_ENDPOINT}")
    click.echo(f"  gemini:       {config.GEMINI_DEFAULT_ENDPOINT}")
    click.echo(f"  custom:       Via request template file (-r)")
    
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
            click.echo(f"  Classifier:      ✅ trained ({mode}, {dims} dims)")
        except Exception:
            click.echo(f"  Classifier:      ✅ trained")
    else:
        click.echo(f"  Classifier:      ❌ not trained  → run 'train'")

    if config.TEMPLATES_PATH.exists():
        tc = TemplateClassifier()
        tc.load(str(config.TEMPLATES_PATH))
        click.echo(f"  Family templates:✅ {len(tc.templates)} families "
                   f"({', '.join(sorted(tc.templates))})")
    else:
        click.echo(f"  Family templates:⚠️  not built   → run 'build-templates'")

    if config.MODEL_TEMPLATES_PATH.exists():
        mtc = TemplateClassifier()
        mtc.load(str(config.MODEL_TEMPLATES_PATH))
        click.echo(f"  Model templates: ✅ {len(mtc.templates)} models "
                   f"({', '.join(sorted(mtc.templates))})")
    else:
        click.echo(f"  Model templates: ⚠️  not built   → run 'build-model-templates'")

    click.echo(f"\n💡 Training options:")
    click.echo(f"  train              # Rebalanced per-layer features (default)")
    click.echo(f"  train --use-pca    # Additional global PCA reduction")


@cli.command()
@backend_options
@click.option('--model', '-m', default=None, help='Model name')
@click.option('--prompt', '-p', default="Hello! How are you?", help='Test prompt')
@click.pass_context
def test(ctx, backend, endpoint, api_key, request_file, model, prompt):
    """Test connection and generation with a backend.
    
    \b
    Examples:
      test -b ollama --model llama3.2
      test -b openai --model gpt-4o-mini
      test -b custom -r ./request.txt
      test -b custom -r ./request.txt -k my-api-key
    """
    print_header()
    
    # Validate and show config
    if backend == "custom":
        if not request_file:
            raise click.ClickException("Custom backend requires --request-file (-r)")
        click.echo(f"🔌 Backend: custom")
        click.echo(f"📄 Request file: {request_file}")
    else:
        endpoint = endpoint or get_default_endpoint(backend)
        click.echo(f"🔌 Backend: {backend}")
        click.echo(f"🌐 Endpoint: {endpoint}")
    
    click.echo(f"🤖 Model: {model or 'from template'}")
    
    try:
        client = get_api_client(backend, endpoint, api_key, request_file)
        
        click.echo(f"\n🔍 Checking connectivity...")
        if not client._check_connectivity():
            click.echo(click.style(f"❌ Cannot connect to API", fg='red'))
            sys.exit(1)
        click.echo("✅ Connected")
        
        click.echo(f"\n💬 Testing generation...")
        click.echo(f"   Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        
        response = client.generate(prompt=prompt, model=model, max_tokens=100)
        
        click.echo(f"\n📝 Response:")
        click.echo(f"   {response[:200]}{'...' if len(response) > 200 else ''}")
        click.echo(f"\n✅ Test successful!")
        
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        sys.exit(1)


@cli.command()
@backend_options
@click.option('--model', '-m', default=None, help='Model name')
@click.option('--repeats', default=1, type=int, help='Prompt repeats (default: 1)')
@click.option('--output', type=click.Path(), help='Output directory')
@click.pass_context
def fingerprint(ctx, backend, endpoint, api_key, request_file, model, repeats, output):
    """Generate fingerprint without classification.
    
    \b
    Examples:
      fingerprint -b ollama --model llama3.2
      fingerprint -b custom -r ./request.txt --output ./my_fingerprints
    """
    print_header()
    logger = ctx.obj['logger']
    
    # Validate options
    if backend == "custom":
        if not request_file:
            raise click.ClickException("Custom backend requires --request-file (-r)")
        click.echo(f"🔌 Backend: custom")
        click.echo(f"📄 Request file: {request_file}")
    else:
        endpoint = endpoint or get_default_endpoint(backend)
        click.echo(f"🔌 Backend: {backend}")
        click.echo(f"🌐 Endpoint: {endpoint}")

    try:
        client = get_api_client(backend, endpoint, api_key, request_file)
        if not client._check_connectivity():
            click.echo(click.style(f"❌ Cannot connect to API", fg='red'))
            sys.exit(1)
        click.echo("✅ Connected")

        click.echo(f"\n🔧 Initializing...")
        suite = PromptSuite()
        extractor = FeatureExtractor()
        classifier = EnsembleClassifier(config.MODEL_FAMILIES)
        fingerprinter = LLMFingerprinter(endpoint if backend != "custom" else "custom",
                                         client, suite, extractor, classifier)

        model_display = model or "default"
        click.echo(f"\n📊 Fingerprinting {model_display}...")
        fp = fingerprinter.fingerprint_model(model, repeats=repeats)

        if fp is None:
            click.echo(click.style("❌ Failed", fg='red'))
            sys.exit(1)

        click.echo(f"\n✅ Generated:")
        click.echo(f"   Dimension: {len(fp['vector'])}")
        click.echo(f"   Queries:   {fp['metadata']['queries_executed']}")
        click.echo(f"   Duration:  {fp['metadata']['duration_seconds']:.1f}s")

        save_dir = output or str(config.RESULTS_DIR)
        store = FingerprintStore(save_dir)
        path = store.save_fingerprint(fp, model_display)
        click.echo(f"\n📁 Saved: {path}")

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        logger.exception("Failed")
        sys.exit(1)


@cli.command('build-templates')
@click.option('--ood-ratio', default=0.80, type=float, show_default=True,
              help='OOD ratio threshold (lower = stricter OOD detection).')
@click.pass_context
def build_templates(ctx, ood_ratio):
    """Build open-set template classifier from training fingerprints.

    Templates let you classify new families without retraining the ensemble —
    just run 'add-family' to register any new model with ≥3 fingerprints.

    \b
    Examples:
      build-templates
      build-templates --ood-ratio 0.75
    """
    print_header()
    logger = ctx.obj['logger']

    try:
        training_data = {}
        for search_dir in [config.TRAINING_DIR, config.FINGERPRINTS_DIR]:
            store = FingerprintStore(str(search_dir))
            click.echo(f"📂 Loading from {search_dir.name}/…")
            data = store.export_for_training()
            for family, vectors in data.items():
                training_data.setdefault(family, []).extend(vectors)

        if not training_data:
            click.echo(click.style("❌ No fingerprints found", fg='red'))
            click.echo("   Run 'simulate' first")
            sys.exit(1)

        click.echo("\n📊 Training data:")
        for fam, vecs in sorted(training_data.items()):
            click.echo(f"   {fam:12s} {len(vecs)} samples")

        click.echo(f"\n🔧 Building templates (OOD ratio threshold: {ood_ratio})…")
        tc = TemplateClassifier(ood_ratio_threshold=ood_ratio)
        if not tc.build(training_data):
            click.echo(click.style("❌ Failed to build templates", fg='red'))
            sys.exit(1)

        tc.save(str(config.TEMPLATES_PATH))
        click.echo(f"\n✅ Built {len(tc.templates)} templates: "
                   f"{', '.join(sorted(tc.templates))}")
        click.echo(f"   Saved to: {config.TEMPLATES_PATH}")
        click.echo("   Use 'add-family' to register new families without retraining.")

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        logger.exception("Failed")
        sys.exit(1)


@cli.command('build-model-templates')
@click.option('--ood-ratio', default=0.80, type=float, show_default=True,
              help='OOD ratio threshold for model-level classifier.')
@click.option('--min-samples', default=1, type=int, show_default=True,
              help='Skip models with fewer than this many fingerprints.')
@click.pass_context
def build_model_templates(ctx, ood_ratio, min_samples):
    """Build per-model templates for specific model version identification.

    Uses the exact model names stored during 'simulate' to build a second
    template classifier. Run this after 'build-templates'.

    \b
    After running this, 'identify' will show a model-level estimate alongside
    the family prediction, e.g.:
        Family:  gpt  (87%)
        Model:   gpt-4o  (74%)

    \b
    Examples:
      build-model-templates
      build-model-templates --min-samples 2
    """
    print_header()
    logger = ctx.obj['logger']

    try:
        model_data = {}
        model_family_map = {}
        for search_dir in [config.TRAINING_DIR, config.FINGERPRINTS_DIR]:
            store = FingerprintStore(str(search_dir))
            click.echo(f"📂 Loading from {search_dir.name}/…")
            data = store.export_by_model()
            for model_name, vectors in data.items():
                model_data.setdefault(model_name, []).extend(vectors)
            fmap = store.export_model_family_map()
            model_family_map.update(fmap)

        if not model_data:
            click.echo(click.style("❌ No fingerprints found", fg='red'))
            click.echo("   Run 'simulate' first")
            sys.exit(1)

        # Filter out models below min-samples threshold
        skipped = [m for m, v in model_data.items() if len(v) < min_samples]
        model_data = {m: v for m, v in model_data.items() if len(v) >= min_samples}

        if skipped:
            click.echo(f"\n⚠️  Skipped (< {min_samples} sample(s)): {', '.join(skipped)}")

        click.echo("\n📊 Model data:")
        for m, vecs in sorted(model_data.items()):
            fam = model_family_map.get(m, '?')
            click.echo(f"   {m:30s}  {len(vecs)} sample(s)  [{fam}]")

        if not model_data:
            click.echo(click.style("❌ No models meet the minimum sample threshold", fg='red'))
            sys.exit(1)

        if model_family_map:
            click.echo(f"\n   Family labels resolved for "
                       f"{len(model_family_map)}/{len(model_data)} models")

        click.echo(f"\n🔬 Building model-level templates ({len(model_data)} models)…")
        mtc = TemplateClassifier(ood_ratio_threshold=ood_ratio)
        if not mtc.build(model_data, model_families=model_family_map):
            click.echo(click.style("❌ Failed to build model templates", fg='red'))
            sys.exit(1)

        mtc.save(str(config.MODEL_TEMPLATES_PATH))
        click.echo(f"\n✅ Built templates for: {', '.join(sorted(mtc.templates))}")
        click.echo(f"   Saved to: {config.MODEL_TEMPLATES_PATH}")
        click.echo("   'identify' will now show a model-level estimate automatically.")

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        logger.exception("Failed")
        sys.exit(1)


@cli.command('add-family')
@backend_options
@click.option('--family', required=True,
              help='Family name to register (e.g. "deepseek").')
@click.option('--model', '-m', default=None, help='Model name on the backend.')
@click.option('--num-fps', default=5, type=int, show_default=True,
              help='Number of fingerprints to collect (≥3 recommended).')
@click.option('--repeats', default=1, type=int, show_default=True,
              help='Prompt repeats per fingerprint.')
@click.pass_context
def add_family(ctx, backend, endpoint, api_key, request_file,
               family, model, num_fps, repeats):
    """Add a new model family to the open-set template classifier.

    Generates fingerprints for the model, computes a class-mean template, and
    saves it alongside existing templates — no ensemble retraining needed.

    \b
    Examples:
      add-family -b ollama -m deepseek-r1 --family deepseek
      add-family -b openai -m gpt-4.1 --family gpt --num-fps 8
      add-family -b custom -r request.txt --family my-model
    """
    print_header()
    logger = ctx.obj['logger']

    if backend == "custom":
        if not request_file:
            raise click.ClickException("Custom backend requires --request-file (-r)")
        click.echo(f"🔌 Backend: custom | 📄 {request_file}")
    else:
        endpoint = endpoint or get_default_endpoint(backend)
        click.echo(f"🔌 Backend: {backend}  🌐 {endpoint}")

    try:
        client = get_api_client(backend, endpoint, api_key, request_file)
        if not client._check_connectivity():
            click.echo(click.style("❌ Cannot connect to API", fg='red'))
            sys.exit(1)
        click.echo("✅ Connected")

        suite = PromptSuite()
        extractor = FeatureExtractor()
        classifier = EnsembleClassifier(config.MODEL_FAMILIES)
        fingerprinter = LLMFingerprinter(
            endpoint if backend != "custom" else "custom",
            client, suite, extractor, classifier
        )

        model_display = model or "default"
        click.echo(f"\n📊 Collecting {num_fps} fingerprints for "
                   f"'{family}' ({model_display})…")

        vectors = []
        for i in range(num_fps):
            click.echo(f"  [{i + 1}/{num_fps}] Fingerprinting…", nl=False)
            fp = fingerprinter.fingerprint_model(model, repeats=repeats)
            if fp is None:
                click.echo(click.style("  ⚠️  failed, skipping", fg='yellow'))
                continue
            vectors.append(fp['vector'])
            click.echo(f"  ✅ ({fp['metadata']['queries_executed']} queries)")

        if not vectors:
            click.echo(click.style("❌ No fingerprints collected", fg='red'))
            sys.exit(1)

        if len(vectors) < 3:
            click.echo(click.style(
                f"⚠️  Only {len(vectors)} fingerprint(s) — recommend ≥3 for a "
                f"reliable template", fg='yellow'
            ))

        # Load or start a fresh template store
        tc = TemplateClassifier()
        if config.TEMPLATES_PATH.exists():
            tc.load(str(config.TEMPLATES_PATH))
            click.echo(f"\n📂 Existing templates: {', '.join(sorted(tc.templates))}")
        else:
            click.echo("\n⚠️  No existing templates — creating new store")

        tc.add_family(family, vectors)
        tc.save(str(config.TEMPLATES_PATH))

        click.echo(f"\n✅ Added '{family}' from {len(vectors)} fingerprints")
        click.echo(f"   All families: {', '.join(sorted(tc.templates))}")
        click.echo(f"   Saved to: {config.TEMPLATES_PATH}")
        click.echo(click.style(
            f"\n   ℹ️  Template-only: '{family}' works for open-set identification,\n"
            f"   but is NOT included in the ensemble classifier.\n"
            f"   To include it in ensemble retraining: add '{family}' to\n"
            f"   MODEL_FAMILIES in config.py, then run 'simulate' + 'train'.",
            fg='cyan'))

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        logger.exception("Failed")
        sys.exit(1)


if __name__ == '__main__':
    cli()