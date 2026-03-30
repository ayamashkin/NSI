"""
Utils Package
Вспомогательные модули для работы с данными.
"""

# НЕ импортируем excel_loader здесь, чтобы избежать импорта pandas
# pandas загружается только при явном вызове функций

from .json_export import JSONExporter, export_results

__all__ = [
    'JSONExporter',
    'export_results'
]
