"""
Tests for Automated Parametric Search System
"""

import unittest
from pathlib import Path
import tempfile
import shutil


class TestMaskDatabase(unittest.TestCase):
    """Тесты для MaskDatabase."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_masks.db"

        from database.mask_database import MaskDatabase
        self.db = MaskDatabase(db_path=str(self.db_path))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.temp_dir)

    def test_save_and_get_mask(self):
        """Тест сохранения и получения маски."""
        from database.mask_database import MaskRecord

        mask = MaskRecord(
            standard="ГОСТ 7798-70",
            item_type="болт",
            pattern=r"Болт\s+M(?P<диаметр>\d+)",
            params=["диаметр"],
            required=["диаметр"],
            auto_score=0.90,
            is_active=True
        )

        mask_id = self.db.save_mask(mask)
        self.assertIsNotNone(mask_id)

        retrieved = self.db.get_mask("ГОСТ 7798-70", "болт")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.standard, "ГОСТ 7798-70")
        self.assertEqual(retrieved.item_type, "болт")

    def test_statistics(self):
        """Тест статистики."""
        stats = self.db.get_statistics()
        self.assertIn('total', stats)
        self.assertIn('active', stats)


class TestStandardExtractor(unittest.TestCase):
    """Тесты для StandardExtractor."""

    def setUp(self):
        from parsers.standard_extractor import StandardExtractor
        self.extractor = StandardExtractor()

    def test_extract_gost(self):
        """Тест извлечения ГОСТ."""
        text = "Болт М12х50 ГОСТ 7798-70"
        result = self.extractor.extract(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.standard_type.value, "ГОСТ")
        self.assertEqual(result.standard_number, "7798")
        self.assertEqual(result.year, "70")

    def test_extract_ost(self):
        """Тест извлечения ОСТ."""
        text = "Болт (2)-12-44-Окс.Фос.ЭФП-ОСТ 1 31133-80"
        result = self.extractor.extract(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.standard_type.value, "ОСТ")

    def test_extract_type(self):
        """Тест извлечения типа изделия."""
        text = "Болт М12х50 ГОСТ 7798-70"
        item_type = self.extractor.extract_type(text)

        self.assertEqual(item_type, "болт")


class TestAutoValidator(unittest.TestCase):
    """Тесты для AutoValidator."""

    def setUp(self):
        from validators.auto_validator import AutoValidator
        self.validator = AutoValidator(min_examples=2)

    def test_validate_mask(self):
        """Тест валидации маски."""
        pattern = r"Болт\s+M(?P<диаметр>\d+)"

        examples = [
            {"полное_наименование": "Болт М12", "диаметр": "12"},
            {"полное_наименование": "Болт М16", "диаметр": "16"},
        ]

        result = self.validator.validate_mask(
            pattern=pattern,
            params=["диаметр"],
            required=["диаметр"],
            standard="ГОСТ 7798-70",
            item_type="болт",
            ens_examples=examples
        )

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)


class TestLLMMaskGenerator(unittest.TestCase):
    """Тесты для LLMMaskGenerator."""

    def test_quality_gate(self):
        """Тест ворот качества."""
        from generators.llm_mask_generator import MaskQualityGate

        gate = MaskQualityGate()

        action, _ = gate.evaluate(0.90)
        self.assertEqual(action, "activate")

        action, _ = gate.evaluate(0.70)
        self.assertEqual(action, "draft")

        action, _ = gate.evaluate(0.40)
        self.assertEqual(action, "reject")


if __name__ == '__main__':
    unittest.main()
