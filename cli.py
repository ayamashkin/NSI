#!/usr/bin/env python3
"""
Nomenclature Processor CLI
LAST_FIX: 2026-05-18 22:15 UTC+3
  - generate-masks: fallback search by case (upper/lower) for existing masks
  - _init_llm_clients: only services from config.yaml (settings.api.keys())
"""

import click
import logging
import threading
import yaml
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime

from config.settings import setup_logging

logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', default='config/config.yaml', help='Path to config')
@click.pass_context
def cli(ctx, config):
    """Nomenclature Processor"""
    ctx.ensure_object(dict)
    config_path = Path(config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            ctx.obj['config'] = yaml.safe_load(f)
    else:
        logger.warning("Config not found: %s", config)
        ctx.obj['config'] = {}
    try:
        setup_logging(str(config_path))
    except Exception as e:
        logger.warning("Failed to setup logging: %s", e)


# ======================================================================
# LEGACY COMMANDS
# ======================================================================

@cli.command()
def prompts():
    """List all available prompts"""
    from config.settings import get_settings
    settings = get_settings()
    click.echo("Available prompts:")
    for pid, cfg in settings.prompts.items():
        click.echo("")
        click.echo(f"  {pid}")
        click.echo(f"    Name: {cfg.name}")
        click.echo(f"    Category: {cfg.category}")
        click.echo(f"    Service: {cfg.resolve_service(settings)}")
        click.echo(f"    Model: {cfg.resolve_model(settings)}")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--prompt', '-p', multiple=True, help='Prompt IDs')
@click.option('--auto', is_flag=True, help='Auto-select prompts')
@click.option('--workers', '-w', default=None, type=int, help='Worker count')
@click.option('--force', '-f', is_flag=True, help='Force reprocess')
@click.pass_context
def process(ctx, input_file, prompt, auto, workers, force):
    """Process nomenclature file (legacy LLM mode)"""
    from config.settings import get_settings
    from core.database import DatabaseManager
    from core.processor import NomenclatureProcessor, load_excel_items
    from utils.excel_loader import ExcelLoader

    settings = get_settings()
    click.echo(f"Processing {input_file}...")
    try:
        items = load_excel_items(input_file)
    except Exception:
        loader = ExcelLoader(input_file)
        items = loader.load()
    click.echo(f"Loaded {len(items)} items")

    db = DatabaseManager(settings.database.path)
    processor = NomenclatureProcessor(db, max_workers=workers)
    prompt_ids = list(prompt) if prompt else []
    if auto:
        click.echo("Auto-selecting prompts...")
        results = processor.auto_process(items, force_reprocess=force)
    else:
        if not prompt_ids:
            click.echo("Specify --auto or --prompt", err=True)
            return
        results = processor.process_batch(items, prompt_ids, force_reprocess=force)
    click.echo(f"Processed: {len(results)} items")


@cli.command()
@click.option('--output', '-o', default='results.json', help='Output file')
@click.option('--structure', type=click.Choice(['flat', 'by_code', 'by_category', 'by_prompt']),
              default='flat', help='Output structure')
@click.option('--prompt', '-p', help='Filter by prompt ID')
@click.option('--status', '-s', help='Filter by status')
@click.option('--include-raw', is_flag=True, help='Include raw response')
@click.option('--include-full-request', is_flag=True, help='Include full request')
def export(output, structure, prompt, status, include_raw, include_full_request):
    """Export results to JSON"""
    from config.settings import get_settings
    from core.database import DatabaseManager
    settings = get_settings()
    db = DatabaseManager(settings.database.path)
    click.echo("Exporting results...")
    results = db.get_all_results(category=None, status=status, prompt_id=prompt, limit=None)
    if not results:
        click.echo("No data to export")
        return
    db.export_filtered_to_json(
        output_path=output, results=results, structure=structure,
        include_raw=include_raw, include_full_request=include_full_request
    )
    click.echo(f"Exported {len(results)} items to {output}")


@cli.command()
def stats():
    """Show processing statistics"""
    from config.settings import get_settings
    from core.database import DatabaseManager
    settings = get_settings()
    db = DatabaseManager(settings.database.path)
    stats = db.get_statistics()
    click.echo(f"Total: {stats.get('total', 0)}")
    for status, count in stats.get('by_status', {}).items():
        click.echo(f"  {status}: {count}")


@cli.command()
@click.option('--limit', '-l', default=10, help='Error count')
@click.option('--prompt', '-p', help='Filter by prompt ID')
def errors(limit, prompt):
    """Show recent errors"""
    from config.settings import get_settings
    from core.database import DatabaseManager
    settings = get_settings()
    db = DatabaseManager(settings.database.path)
    error_results = db.get_all_results(status='error', prompt_id=prompt, limit=limit)
    if not error_results:
        click.echo("No errors found")
        return
    click.echo(f"Found {len(error_results)} errors:")
    for i, result in enumerate(error_results, 1):
        click.echo(f"{i}. {result.get('article', 'N/A')}: {result.get('name', 'N/A')[:50]}...")
        click.echo(f"   Error: {result.get('error_message', 'N/A')[:100]}...")


# ===================================================================
# PARAMETRIC COMMANDS
# ===================================================================

def _init_llm_clients(settings, all_services=False):
    """Initialize LLM clients. Only services from config.yaml."""
    llm_clients = {}
    if all_services:
        services = list(settings.api.keys())
        logger.info("LLM: initializing all configured services: %s", services)
    else:
        services = [settings.mask_generation.default_service]
        logger.info("LLM: using default_service='%s'", services[0])

    for service_name in services:
        if service_name not in settings.api:
            logger.warning("Service '%s' not found in settings.api, skipping", service_name)
            continue
        try:
            cfg = settings.api[service_name]
            if service_name == 'openwebui':
                if not cfg.api_key and not (cfg.username and cfg.password):
                    logger.debug("Skipping %s: no credentials", service_name)
                    continue
                from api_clients.openwebui import OpenWebUIClient
                llm_clients[service_name] = OpenWebUIClient(
                    base_url=cfg.base_url, api_key=cfg.api_key,
                    username=cfg.username, password=cfg.password
                )
            elif service_name == 'mws':
                if not cfg.api_key:
                    logger.debug("Skipping %s: no api_key", service_name)
                    continue
                from api_clients.mws_gpt import MWSGPTClient
                llm_clients[service_name] = MWSGPTClient(
                    base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout
                )
            elif service_name == 'gigachat':
                if not cfg.api_key:
                    logger.debug("Skipping %s: no api_key", service_name)
                    continue
                from api_clients.gigachat import GigaChatClient
                llm_clients[service_name] = GigaChatClient(
                    base_url=cfg.base_url, api_key=cfg.api_key,
                    scope=getattr(cfg, 'scope', 'GIGACHAT_API_PERS'),
                    timeout=cfg.timeout, verify_ssl=False
                )
            elif service_name == 'mts_ai':
                if not cfg.api_key:
                    logger.debug("Skipping %s: no api_key", service_name)
                    continue
                from api_clients.mts_ai import MTSAIClient
                llm_clients[service_name] = MTSAIClient(
                    base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout
                )
            else:
                if not getattr(cfg, 'api_key', None):
                    logger.debug("Skipping %s: no api_key", service_name)
                    continue
                logger.warning("Unknown service '%s', skipping", service_name)
                continue
            logger.info("LLM client initialized: %s", service_name)
        except Exception as e:
            logger.warning("Failed to init %s: %s", service_name, e)
    return llm_clients


@cli.command()
@click.argument('text')
@click.option('--db', '-d', default='cache/masks.db', help='Mask DB path')
@click.option('--ens-index', '-i', required=True, help='ENS index path')
@click.option('--llm', '-l', is_flag=True, help='Enable LLM generation')
def process_parametric(text, db, ens_index, llm):
    """Process single item via parametric search"""
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from config.settings import get_settings

    llm_clients = {}
    settings = get_settings()
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("LLM requested but no clients available", err=True)
            return
        click.echo("LLM generation enabled")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )
    result = processor.process(text)
    click.echo(f"Text: {result.text}")
    click.echo(f"Level: {result.level.value}")
    click.echo(f"Success: {result.success}")
    click.echo(f"Confidence: {result.confidence:.2f}")
    if result.params:
        click.echo("Params:")
        for key, value in result.params.items():
            if not key.startswith('_'):
                click.echo(f"  {key}: {value}")
    if result.ens_match:
        click.echo(f"ENS code: {result.ens_match.get('code')}")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--db', '-d', default='cache/masks.db', help='Mask DB path')
@click.option('--ens-index', '-i', required=True, help='ENS index path')
@click.option('--output', '-o', default='results.json', help='Output file')
@click.option('--llm', '-l', is_flag=True, help='Enable LLM generation')
@click.option('--validate/--no-validate', default=True, help='Auto validation')
@click.option('--success-only', is_flag=True, help='Only successful results')
@click.option('--include-details', is_flag=True, help='Include debug details')
@click.option('--coating-map', '-c', help='Coating mapping Excel file')
@click.option('--workers', '-w', default=4, type=int, help='Number of parallel workers (default: 4)')
def batch(input_file, db, ens_index, output, llm, validate, success_only, include_details, coating_map, workers):
    """Batch process nomenclature file"""
    import pandas as pd
    from tqdm import tqdm
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from config.settings import get_settings

    click.echo(f"Processing {input_file}...")
    df = pd.read_excel(input_file)

    name_col = 'Full name'
    if name_col not in df.columns:
        name_cols = [c for c in df.columns if 'name' in str(c).lower()]
        if name_cols:
            name_col = name_cols[0]
        else:
            click.echo("Name column not found", err=True)
            return

    texts = df[name_col].astype(str).tolist()
    click.echo(f"Loaded {len(texts)} items")

    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"Coating mapping loaded: {coating_map}")

    settings = get_settings()
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("LLM requested but no clients available", err=True)
            return
        click.echo("LLM generation enabled")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    click.echo("Processing items...")
    click.echo(f"Workers: {workers}")
    if success_only:
        click.echo("Filter: success only")

    results = [None] * len(texts)
    stats = {'total': 0, 'success': 0, 'failed': 0, 'filtered': 0}
    stats_lock = threading.Lock()

    def _process_one(idx_text):
        idx, text = idx_text
        try:
            result = processor.process(text)
            with stats_lock:
                nonlocal stats
                stats['total'] += 1
                if result.success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                if success_only and not result.success:
                    stats['filtered'] += 1
                    return idx, None

            row = {
                'text': result.text,
                'level': result.level.value,
                'success': result.success,
                'params': result.params,
                'ens_code': result.ens_match.get('code') if result.ens_match else None,
                'ens_name': result.ens_match.get('name') if result.ens_match else None,
                'ens_params': result.ens_params,
                'ens_params_mask': result.ens_params_mask,
                'confidence': result.confidence,
                'processing_time_ms': result.processing_time_ms,
                'item_type': result.item_type,
                'standard': result.standard,
                'mask_pattern': result.details.get('mask_pattern') if result.details else None,
                'match_type': result.details.get('match_type') if result.details else None,
                'match_type_ru': result.details.get('match_type_ru') if result.details else None,
                'coating_substitution': result.details.get('coating_substitution') if result.details else None,
                'fuzzy_mismatched_params': result.details.get('fuzzy_mismatched_params') if result.details else None,
                'fuzzy_params_comparison': result.details.get('fuzzy_params_comparison') if result.details else None,
            }
            if include_details and result.details:
                row['details'] = result.details
            return idx, row
        except Exception as e:
            logger.error("Error processing item %d: %s", idx, e)
            with stats_lock:
                stats['total'] += 1
                stats['failed'] += 1
            return idx, None

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, (i, t)): i for i, t in enumerate(texts)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(texts), desc="Processing"):
            idx, row = future.result()
            if row is not None:
                results[idx] = row

    # Filter out None entries (filtered or failed)
    results = [r for r in results if r is not None]

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    click.echo("")
    click.echo(f"Results: {output} ({len(results)} items)")
    click.echo(f"Total: {stats['total']}")
    click.echo(f"Success: {stats['success']}")
    click.echo(f"Failed: {stats['failed']}")
    if success_only:
        click.echo(f"Filtered: {stats['filtered']}")


@cli.command('analyze-quality')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--db', '-d', default='cache/masks.db', help='Mask DB path')
@click.option('--ens-index', '-i', required=True, help='ENS index path')
@click.option('--output', '-o', help='Excel report file')
@click.option('--json', '-j', 'json_output', help='JSON report file')
@click.option('--llm', '-l', is_flag=True, help='Enable LLM generation')
@click.option('--coating-map', '-c', help='Coating mapping Excel file')
def analyze_quality_cmd(input_file, db, ens_index, output, json_output, llm, coating_map):
    """Analyze recognition quality"""
    from core.quality_analyzer import QualityAnalyzer
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from config.settings import get_settings

    settings = get_settings()
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("LLM requested but no clients available", err=True)
            return
        click.echo("LLM generation enabled")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"Coating mapping loaded: {coating_map}")

    analyzer = QualityAnalyzer(processor=processor)
    click.echo(f"Analyzing quality: {input_file}...")
    stats = analyzer.analyze_file(input_file)
    report_text = analyzer.format_report(stats)
    click.echo("")
    click.echo(report_text)

    if output:
        analyzer.save_excel(stats, output)
        click.echo("")
        click.echo(f"Excel report: {output}")

    if json_output:
        analyzer.save_json(stats, json_output)
        click.echo(f"JSON report: {json_output}")


@cli.command('diagnose')
@click.argument('text')
@click.option('--db', '-d', default='cache/masks.db', help='Mask DB path')
@click.option('--ens-index', '-i', required=True, help='ENS index path')
@click.option('--llm', '-l', is_flag=True, help='Enable LLM generation')
@click.option('--coating-map', '-c', help='Coating mapping Excel file')
def diagnose(text, db, ens_index, llm, coating_map):
    """Diagnose single item processing"""
    import re
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from core.parametric_client import ParametricENSClient
    from config.settings import get_settings

    settings = get_settings()
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("LLM requested but no clients available", err=True)
            return
        click.echo("LLM generation enabled")

    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"Coating mapping loaded: {coating_map}")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    click.echo("")
    click.echo("=" * 60)
    click.echo(f"DIAGNOSE: {text}")
    click.echo("=" * 60)

    # Step 0
    extracted = processor.standard_extractor.extract_all(text)
    standard_info = extracted.get('standard_info')
    item_type = extracted.get('item_type')
    click.echo("")
    click.echo("Step 0 (Standard extraction):")
    click.echo(f"   standard_info: {standard_info.to_dict() if standard_info else None}")
    click.echo(f"   item_type: {item_type}")

    if not standard_info or not item_type:
        click.echo("")
        click.echo("Could not extract standard or type")
        return

    standard = standard_info.normalized
    search_item_type = item_type.upper()

    # Step 1
    click.echo("")
    click.echo("Step 1 (Mask lookup):")
    mask = mask_db.get_mask(standard, search_item_type)
    click.echo(f"   Search: standard='{standard}', item_type='{search_item_type}'")
    click.echo(f"   Found: {mask is not None}")

    if mask is None:
        mask = mask_db.get_mask(standard, item_type)
        if mask:
            click.echo(f"   Found (case): item_type='{item_type}'")

    if mask is None:
        click.echo("   Mask not found in DB")
        return

    click.echo(f"   mask.id: {getattr(mask, 'id', 'N/A')}")
    click.echo(f"   mask.standard: {getattr(mask, 'standard', 'N/A')}")
    click.echo(f"   mask.item_type: {getattr(mask, 'item_type', 'N/A')}")
    click.echo(f"   mask.is_active: {getattr(mask, 'is_active', 'N/A')}")
    click.echo(f"   mask.pattern (first 120 chars):")
    click.echo(f"      {getattr(mask, 'pattern', 'N/A')[:120]}")

    # Step 2
    effective_standard = getattr(mask, 'standard', None) or standard
    client = ParametricENSClient.__new__(ParametricENSClient)
    relaxed = client._relax_pattern(mask.pattern, standard=effective_standard)
    click.echo("")
    click.echo("Step 2 (Relax pattern):")
    click.echo(f"   standard for relax: '{effective_standard}'")
    click.echo(f"   relaxed (first 200 chars):")
    click.echo(f"      {relaxed[:200]}")
    if len(relaxed) > 200:
        click.echo(f"      ... (total {len(relaxed)} chars)")

    # Step 3
    try:
        compiled = re.compile(relaxed, re.IGNORECASE)
        match = compiled.search(text)
        click.echo("")
        click.echo("Step 3 (Regex match):")
        if match:
            click.echo("   MATCH")
            click.echo(f"   groups: {match.groupdict()}")
        else:
            click.echo("   NO MATCH")
            for i in range(len(text), 0, -1):
                if compiled.search(text[:i]):
                    click.echo(f"   longest matching prefix: '{text[:i]}'")
                    break
            else:
                click.echo("   no prefix matches at all")
    except re.error as e:
        click.echo("")
        click.echo("Step 3 (Regex match):")
        click.echo(f"   INVALID REGEX: {e}")
        click.echo(f"   pattern: {relaxed[:100]}")

    # Step 4
    click.echo("")
    click.echo("Step 4 (Full processor result):")
    result = processor.process(text)
    click.echo(f"   level: {result.level.value}")
    click.echo(f"   success: {result.success}")
    click.echo(f"   params: {result.params}")
    click.echo(f"   ens_code: {result.ens_match.get('code') if result.ens_match else None}")
    click.echo(f"   ens_params: {result.ens_params}")
    click.echo(f"   confidence: {result.confidence:.3f}")
    click.echo(f"   processing_time_ms: {result.processing_time_ms:.1f}")
    if result.details:
        click.echo(f"   details: {result.details}")

    click.echo("")
    click.echo("=" * 60)


@cli.group()
def ens():
    """ENS index operations"""
    pass


@ens.command('auto-mapping')
@click.argument('excel_file', type=click.Path(exists=True))
@click.option('--output', '-o', required=True, help='Output YAML path')
@click.option('--append', is_flag=True, help='Append to existing YAML')
def auto_mapping(excel_file, output, append):
    """Generate ens_column_mapping.yaml from Excel"""
    from auto_mapping import generate_mapping
    import yaml
    click.echo(f"Generating mapping from {excel_file}...")
    mapping = generate_mapping(excel_file, append=append, existing_yaml=output if append else None)
    with open(output, 'w', encoding='utf-8') as f:
        yaml.dump(mapping, f, allow_unicode=True, sort_keys=False)
    total = sum(len(v) for v in mapping.get('category_mapping', {}).values())
    click.echo(f"Mapped {total} columns: {output}")


@ens.command()
@click.argument('excel_file', type=click.Path(exists=True))
@click.option('--output', '-o', required=True, help='Output .pkl path')
@click.option('--category', '-c', type=click.Choice(['hardware', 'washer', 'rolledmetal']))
def build_index(excel_file, output, category):
    """Build ENS index from Excel"""
    from core.integration import build_ens_index
    click.echo(f"Building index from {excel_file}...")
    result_path = build_ens_index(excel_file, output, category)
    click.echo(f"Index saved: {result_path}")
    meta_path = Path(result_path).with_suffix('.meta.json')
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        click.echo(f"Items: {meta.get('item_count', 0)}")
        click.echo(f"Category: {meta.get('category', 'unknown')}")


@ens.command()
@click.argument('query')
@click.option('--index', '-i', required=True, help='Index path')
@click.option('--top-k', '-k', default=5, help='Number of results')
def search(query, index, top_k):
    """Search ENS index"""
    from ens.indexer import ENSIndex
    if not Path(index).exists():
        click.echo(f"Index not found: {index}", err=True)
        return
    ens_index = ENSIndex.load(index)
    results = ens_index.search(query, k=top_k)
    for i, item in enumerate(results, 1):
        score = item.get('_similarity_score', 0)
        name = item.get('name') or item.get('name', 'N/A')
        click.echo(f"{i}. [{score:.2f}] {name[:60]}...")


@ens.command()
@click.argument('excel_file', type=click.Path(exists=True))
@click.option('--index', '-i', required=True, help='Index path')
@click.option('--sample', '-s', default=100, help='Sample size')
def analyze(excel_file, index, sample):
    """Analyze nomenclature quality"""
    from core.integration import analyze_nomenclature
    stats = analyze_nomenclature(excel_file, index, sample_size=sample)
    click.echo(f"Analysis (sample {sample}):")
    click.echo(f"   Regex parsed: {stats.get('regex_parsed', 0)} ({stats.get('regex_parsed', 0) / sample * 100:.1f}%)")
    click.echo(f"   Failed (need LLM): {stats.get('failed', 0)} ({stats.get('failed', 0) / sample * 100:.1f}%)")
    if 'estimated_regex_parsed' in stats:
        total = stats.get('total', 0)
        click.echo("")
        click.echo(f"Estimated for all {total} items:")
        click.echo(f"   Regex: {stats['estimated_regex_parsed']}")


@cli.command()
@click.option('--db', '-d', default='cache/masks.db', help='Mask DB path')
@click.option('--ens-index', '-i', required=True, help='ENS index path')
@click.option('--llm', '-l', is_flag=True, help='Enable LLM generation')
@click.option('--min-score', '-s', default=0.85, help='Min score for activation')
@click.option('--limit', '-n', default=0, help='Limit number of standards (0 = all)')
@click.option('--standard', help='Filter by specific standard')
@click.option('--force', '-f', is_flag=True, help='Force regeneration even if mask exists')
def generate_masks(db, ens_index, llm, min_score, limit, standard, force):
    """Generate masks from ENS index (incremental fill)"""
    import pickle
    from core.mask_database import MaskDatabase
    from generators.llm_mask_generator import LLMMaskGenerator
    from config.settings import get_settings

    click.echo(f"Loading index: {ens_index}")
    with open(ens_index, 'rb') as f:
        ens_data = pickle.load(f)

    standards: Dict[Tuple[str, str], list] = {}
    for item in ens_data:
        std = item.get('standard', item.get('стандарт', ''))
        itype = item.get('item_type', item.get('тип_изделия', item.get('наименование_типа', '')))
        if std and itype:
            key = (std, itype)
            standards.setdefault(key, []).append(item)

    if standard:
        standards = {k: v for k, v in standards.items() if k[0] == standard}
        click.echo(f"Filter by standard: {standard} ({len(standards)} groups)")

    if limit > 0:
        standards = dict(list(standards.items())[:limit])
        click.echo(f"Limit: {limit} standards")

    click.echo(f"Found {len(standards)} unique (standard, type)")

    mask_db = MaskDatabase(db_path=db)
    stats = {'existing': 0, 'generated': 0, 'failed': 0, 'skipped': 0}

    generator = None
    if llm:
        settings = get_settings()
        llm_clients = _init_llm_clients(settings, all_services=True)
        if llm_clients:
            generator = LLMMaskGenerator(
                clients=llm_clients,
                settings=settings,
                max_retries=3
            )
            click.echo("LLM generator initialized")
        else:
            click.echo("LLM not available — statistics only")

    with click.progressbar(standards.items(), label='Generating') as bar:
        for (std, item_type), examples in bar:
            # === INCREMENTAL: check mask in DB with case fallback ===
            if not force:
                existing = mask_db.get_mask(std, item_type)
                if not existing:
                    existing = mask_db.get_mask(std, item_type.upper())
                if not existing:
                    existing = mask_db.get_mask(std, item_type.lower())
                if existing and existing.is_active:
                    stats['existing'] += 1
                    continue
            else:
                # Force mode: deactivate existing mask before regeneration
                existing = mask_db.get_mask(std, item_type)
                if not existing:
                    existing = mask_db.get_mask(std, item_type.upper())
                if not existing:
                    existing = mask_db.get_mask(std, item_type.lower())
                if existing:
                    try:
                        mask_db.deactivate_mask(existing.id)
                        click.echo(f"   Deactivated existing mask {existing.id} for {std}/{item_type}")
                    except Exception as e:
                        logger.warning("Failed to deactivate mask %s: %s", existing.id, e)

            if generator:
                try:
                    mask, _ = generator.generate_mask(
                        standard=std,
                        item_type=item_type,
                        examples=examples
                    )
                    if mask:
                        mask_record = mask_db.MaskRecord(
                            standard=std,
                            item_type=item_type,
                            pattern=mask['pattern'],
                            params=mask['params'],
                            required=mask['required'],
                            auto_score=0.9,
                            is_active=True,
                            source='llm'
                        )
                        mask_db.save_mask(mask_record)
                        stats['generated'] += 1
                    else:
                        stats['failed'] += 1
                except Exception as e:
                    logger.warning("Failed to generate mask for %s/%s: %s", std, item_type, e)
                    stats['failed'] += 1
            else:
                stats['skipped'] += 1

    click.echo("")
    click.echo("Results:")
    click.echo(f"   Existing (skipped): {stats['existing']}")
    click.echo(f"   Generated: {stats['generated']}")
    click.echo(f"   Failed: {stats['failed']}")
    click.echo(f"   Skipped (no LLM): {stats['skipped']}")


if __name__ == '__main__':
    cli()