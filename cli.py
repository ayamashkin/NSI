"""
CLI интерфейс для управления масками и валидации.
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.domain_config import DomainConfig
from core.ens_index_builder import EnsIndexBuilder
from core.llm_mask_generator import LLMMaskGenerator
from core.gost_normalizer import GOSTNormalizer
from core.automated_processor import AutomatedParametricProcessor
from core.auto_validator import MaskValidator


def cmd_build_index(args):
    """Строит ENS-индекс из Excel."""
    builder = EnsIndexBuilder(args.input)
    records = builder.build()
    builder.save_json(args.output)
    print(f"Индекс построен: {len(records)} записей → {args.output}")


def cmd_generate_masks(args):
    """Генерирует маски для стандартов."""
    generator = LLMMaskGenerator()
    domain = DomainConfig.from_yaml(args.domain) if args.domain else None

    # Загружаем индекс
    with open(args.index, "r", encoding="utf-8") as f:
        index = json.load(f)

    # Группируем по типам и стандартам
    standards = {}
    for r in index:
        itype = r.get("_meta", {}).get("item_type", "")
        std = r.get("_meta", {}).get("standard", "")
        if itype and std:
            key = f"{itype}_{std}"
            standards.setdefault(key, []).append(r)

    masks = {}
    for key, records in standards.items():
        itype, std = key.split("_", 1)
        examples = [r.get("_meta", {}).get("name", "") for r in records[:5]]
        mask = generator.generate(itype, std, examples)
        mask = generator.fix_pattern(mask)
        ok, err = generator.validate(mask)
        if ok:
            masks[key] = mask
            print(f"[OK] {key}")
        else:
            print(f"[ERR] {key}: {err}")

    # Сохраняем
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(masks, f, ensure_ascii=False, indent=2)
    print(f"Маски сохранены: {len(masks)} → {args.output}")


def cmd_validate(args):
    """Валидирует маски."""
    with open(args.index, "r", encoding="utf-8") as f:
        index = json.load(f)
    with open(args.masks, "r", encoding="utf-8") as f:
        masks = json.load(f)

    domain = DomainConfig.from_yaml(args.domain) if args.domain else None
    validator = MaskValidator(index, domain)

    # Готовим тестовые данные
    test_data = {}
    for r in index:
        itype = r.get("_meta", {}).get("item_type", "")
        std = r.get("_meta", {}).get("standard", "")
        key = f"{itype}_{std}" if itype and std else ""
        if key:
            test_data.setdefault(key, []).append({
                "name": r.get("_meta", {}).get("name", ""),
                "expected": r.get("_visible", {}),
            })

    results = validator.validate_all(masks, test_data)
    for key, score in sorted(results.items(), key=lambda x: x[1]):
        status = "OK" if score >= 0.7 else "LOW" if score > 0 else "FAIL"
        print(f"[{status}] {key}: {score:.3f}")


def cmd_process(args):
    """Обрабатывает номенклатуру."""
    with open(args.index, "r", encoding="utf-8") as f:
        index = json.load(f)
    with open(args.masks, "r", encoding="utf-8") as f:
        masks = json.load(f)

    domain = DomainConfig.from_yaml(args.domain) if args.domain else None
    processor = AutomatedParametricProcessor(index, domain, masks)

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    else:
        lines = sys.stdin.read().strip().split("\n")

    results = []
    for line in lines:
        result = processor.process(line)
        results.append({"input": line, "result": result})
        if result:
            print(f"[MATCH] {line} → {result.get('наименование_ens', '?')}")
        else:
            print(f"[SKIP] {line}")

    # Сохраняем результаты
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    stats = processor.get_stats()
    print(f"\nСтатистика: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Параметрический сопоставитель")
    subparsers = parser.add_subparsers(dest="command")

    # build-index
    p = subparsers.add_parser("build-index", help="Построить ENS-индекс")
    p.add_argument("--input", required=True, help="Excel-файл ENS")
    p.add_argument("--output", default="ens_index.json", help="Выходной JSON")
    p.set_defaults(func=cmd_build_index)

    # generate-masks
    p = subparsers.add_parser("generate-masks", help="Сгенерировать маски")
    p.add_argument("--index", required=True, help="JSON-индекс")
    p.add_argument("--domain", help="YAML-конфиг домена")
    p.add_argument("--output", default="masks.json", help="Выходной JSON")
    p.set_defaults(func=cmd_generate_masks)

    # validate
    p = subparsers.add_parser("validate", help="Валидировать маски")
    p.add_argument("--index", required=True, help="JSON-индекс")
    p.add_argument("--masks", required=True, help="JSON с масками")
    p.add_argument("--domain", help="YAML-конфиг домена")
    p.set_defaults(func=cmd_validate)

    # process
    p = subparsers.add_parser("process", help="Обработать номенклатуру")
    p.add_argument("--index", required=True, help="JSON-индекс")
    p.add_argument("--masks", required=True, help="JSON с масками")
    p.add_argument("--domain", help="YAML-конфиг домена")
    p.add_argument("--input", help="Файл с номенклатурой (или stdin)")
    p.add_argument("--output", help="Файл результатов")
    p.set_defaults(func=cmd_process)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()