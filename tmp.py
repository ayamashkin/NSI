#!/usr/bin/env python3
"""
apply_fixes.py
Применяет исправления к cli.py и core/automated_processor.py
для добавления --no-cache и корректного сохранения ens_code.

Запуск:
    python apply_fixes.py
"""

import sys, os

def patch_file(path, replacements, backup=True):
    if not os.path.exists(path):
        print(f"SKIP: {path} not found")
        return False
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    original = content
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  PATCHED: {old[:60]}... -> {new[:60]}...")
        else:
            print(f"  NOT FOUND (may already be patched): {old[:60]}...")
    if content == original:
        print(f"NO CHANGES: {path}")
        return False
    if backup:
        bak = path + '.bak'
        with open(bak, 'w', encoding='utf-8') as f:
            f.write(original)
        print(f"  BACKUP: {bak}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"SAVED: {path}")
    return True

def main():
    print("="*60)
    print("Applying fixes to NSI project")
    print("="*60)

    # 1. cli.py
    cli_replacements = [
        (
            """@click.option('--result-db', '-r', default='cache/result.db', help='Путь к SQLite БД результатов')
@click.option('--domain', default='hardware', help='Домен ENS')""",
            """@click.option('--result-db', '-r', default='cache/result.db', help='Путь к SQLite БД результатов')
@click.option('--no-cache', is_flag=True, help='Пропустить кэш result.db (переобработать все)')
@click.option('--domain', default='hardware', help='Домен ENS')"""
        ),
        (
            """    processor = AutomatedParametricProcessor(
        mask_db=mask_db, llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index, use_llm_generation=llm,
        settings=settings, result_db_path=result_db
    )""",
            """    processor = AutomatedParametricProcessor(
        mask_db=mask_db, llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index, use_llm_generation=llm,
        settings=settings, result_db_path=result_db,
        no_cache=args.no_cache
    )"""
        ),
    ]
    print("\n[1/2] Patching cli.py...")
    patch_file('cli.py', cli_replacements)

    # 2. core/automated_processor.py
    ap_replacements = [
        (
            """    def __init__(self, mask_db: Union[MaskDatabase, str], llm_clients: Optional[Dict] = None,
                 ens_index_path: Optional[str] = None, use_llm_generation: bool = False,
                 settings: Optional[Dict] = None, result_db_path: Optional[str] = None,
                 debug: bool = False, debug_per_parameter: bool = False,
                 debug_prompts_dir: Optional[str] = None, domain: Optional[str] = None):""",
            """    def __init__(self, mask_db: Union[MaskDatabase, str], llm_clients: Optional[Dict] = None,
                 ens_index_path: Optional[str] = None, use_llm_generation: bool = False,
                 settings: Optional[Dict] = None, result_db_path: Optional[str] = None,
                 debug: bool = False, debug_per_parameter: bool = False,
                 debug_prompts_dir: Optional[str] = None, domain: Optional[str] = None,
                 no_cache: bool = False):"""
        ),
        (
            """        self.result_db_path = result_db_path
        self.result_db = None
        self.debug = debug
        self.debug_per_parameter = debug_per_parameter
        self.debug_prompts_dir = debug_prompts_dir""",
            """        self.result_db_path = result_db_path
        self.result_db = None
        self.no_cache = no_cache
        self.debug = debug
        self.debug_per_parameter = debug_per_parameter
        self.debug_prompts_dir = debug_prompts_dir"""
        ),
        (
            """        if not self.result_db_path:
            return []

        if not force:
            # Try to load from cache
            try:
                conn = sqlite3.connect(self.result_db_path)""",
            """        if not self.result_db_path:
            return []

        if not force and not self.no_cache:
            # Try to load from cache
            try:
                conn = sqlite3.connect(self.result_db_path)"""
        ),
    ]
    print("\n[2/2] Patching core/automated_processor.py...")
    patch_file('core/automated_processor.py', ap_replacements)

    print("\n" + "="*60)
    print("Done. Run your command with --no-cache to bypass old results:")
    print("  python cli.py batch data/nomenclature.xlsx ... --no-cache")
    print("="*60)

if __name__ == '__main__':
    main()