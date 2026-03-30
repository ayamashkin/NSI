import yaml
from pathlib import Path
from typing import Dict, List, Optional
from core.models import PromptConfig, Category


class PromptRegistry:
    def __init__(self, config_path: str = "config/prompts.yaml"):
        self.config_path = Path(config_path)
        self.prompts: Dict[str, PromptConfig] = {}
        self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        for prompt_id, p in config['prompts'].items():
            # Загружаем текст промпта из файла
            file_path = Path(p['file'])
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as pf:
                    prompt_text = pf.read()
            else:
                prompt_text = ""

            self.prompts[prompt_id] = PromptConfig(
                id=prompt_id,
                name=p['name'],
                file_path=str(file_path),
                category=Category(p['category']),
                keywords=p['keywords'],
                model=p['model'],
                temperature=p.get('temperature', 0.1),
                prompt_text=prompt_text
            )

    def get(self, prompt_id: str) -> Optional[PromptConfig]:
        return self.prompts.get(prompt_id)

    def list_all(self) -> List[PromptConfig]:
        return list(self.prompts.values())

    def get_by_category(self, category: Category) -> List[PromptConfig]:
        return [p for p in self.prompts.values() if p.category == category]

    def detect_category(self, name: str) -> Optional[Category]:
        """Автоопределение категории по ключевым словам"""
        name_lower = name.lower()

        for prompt_id, config in self.prompts.items():
            for keyword in config.keywords:
                if keyword.lower() in name_lower:
                    return config.category

        return None

    def get_suitable_prompts(self, name: str) -> List[str]:
        """Получить список подходящих промптов для номенклатуры"""
        category = self.detect_category(name)
        if category:
            return [p.id for p in self.get_by_category(category)]
        return []

    def build_prompt(self, prompt_id: str, nomenclature_name: str) -> str:
        """Сформировать полный промпт для отправки в LLM"""
        config = self.get(prompt_id)
        if not config:
            raise ValueError(f"Prompt {prompt_id} not found")

        # Заменяем placeholder на реальное наименование
        prompt = config.prompt_text.replace(
            '"Болт 2M12x1,25-6gx100.58 ГОСТ 7795-70"',
            f'"{nomenclature_name}"'
        )
        return prompt