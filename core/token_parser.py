# =============================================================================
# ФАЙЛ: core/token_parser.py
# ПОСЛЕДНИЕ 5 ИЗМЕНЕНИЙ (МСК, UTC+3):
# 2026-06-01 22:00:00 — СОЗДАНИЕ: TokenParser — гибридный парсер, fallback для regexp
# 2026-06-01 22:00:00 — FEAT: токенизация по разделителям, подсчёт параметров
# 2026-06-01 22:00:00 — FEAT: обработка скобок (исполнение), x-разделитель (шаг)
# =============================================================================

import re
from typing import Dict, List, Optional, Set


class TokenParser:
    """Hybrid parser: tokenizes by separators, assigns params by position count.
    
    Used as fallback when regex fails to match. Handles:
    - Bare execution: 'Винт 1-6-14' (no parentheses)
    - Parenthesized execution: 'Винт (1)-6-14'
    - Optional execution: 'Винт 5-10' (2 numbers = no execution)
    - Thread pitch: '14x1.5' (diameter x pitch)
    - Decimal values: '2,5-24' (comma as decimal separator)
    
    Algorithm:
    1. Strip standard and item type
    2. Extract parenthesized execution
    3. Tokenize by '-' and expand 'x' separated values
    4. Count numeric tokens vs expected params
    5. Assign optional params only when extra tokens exist
    """
    
    STD_PATTERNS = [
        r'ОСТ\s*1\s*\d+-\d+',
        r'ГОСТ\s*\d+-\d+',
        r'ТУ\s*\d+-\d+',
    ]
    
    def parse(self, text: str, standard: str, item_type: str,
              param_order: List[str], optional_params: Set[str]) -> Optional[Dict]:
        """Parse text into parameters. Returns dict or None if parsing fails."""
        if not text or not param_order:
            return None
        
        # Step 1: Strip standard designation from end
        name_part = self._strip_standard(text, standard)
        if not name_part:
            return None
        
        # Step 2: Strip item type (Винт, Болт, etc.)
        if item_type and item_type.lower() in name_part.lower():
            name_part = re.sub(
                rf'\b{item_type}\b', '', name_part,
                count=1, flags=re.IGNORECASE
            ).strip()
        
        # Step 3: Extract parenthesized execution number
        execution = None
        m = re.search(r'\((\d+)\)', name_part)
        if m:
            execution = m.group(1)
            name_part = name_part.replace(m.group(0), '', 1).strip()
        
        # Step 4: Tokenize by '-' and spaces, expand 'x'-separated values
        tokens = re.split(r'[-\s]+', name_part)
        tokens = [t for t in tokens if t]
        
        expanded = []
        for t in tokens:
            # Handle diameter x pitch format: "14x1.5" → ["14", "1.5"]
            if re.match(r'^\d+(?:[.,]\d+)?[xXхХ×]\d+(?:[.,]\d+)?$', t):
                expanded.extend(re.split(r'[xXхХ×]', t))
            else:
                expanded.append(t.strip('()'))
        
        # Step 5: Separate numeric and text tokens
        numeric = []
        text_tokens = []
        for t in expanded:
            if self._is_numeric(t):
                numeric.append(self._normalize(t))
            elif t:
                text_tokens.append(t)
        
        # Step 6: Determine which optional params to use based on token count
        numeric_params = [p for p in param_order 
                         if p not in ['покрытие', 'тип_шлица']]
        required_numeric = [p for p in numeric_params 
                           if p not in optional_params]
        optional_numeric = [p for p in numeric_params 
                           if p in optional_params]
        
        num_count = len(numeric)
        req_count = len(required_numeric)
        opt_count = len(optional_numeric)
        
        # If execution was in parens, don't count it from tokens
        if execution is not None and 'исполнение' in optional_numeric:
            opt_count -= 1
        
        # How many optional params can we fill?
        use_optional = max(0, num_count - req_count)
        use_optional = min(use_optional, opt_count)
        
        # Step 7: Build result
        result = {}
        num_idx = 0
        
        for param in param_order:
            if param == 'исполнение':
                if execution is not None:
                    result[param] = execution
                elif param in optional_numeric and use_optional > 0:
                    if num_idx < len(numeric):
                        result[param] = numeric[num_idx]
                        num_idx += 1
                        use_optional -= 1
            elif param in ['покрытие', 'тип_шлица']:
                if text_tokens:
                    result[param] = text_tokens.pop(0)
            elif param in optional_numeric:
                if use_optional > 0 and num_idx < len(numeric):
                    result[param] = numeric[num_idx]
                    num_idx += 1
                    use_optional -= 1
            else:
                # Required numeric param
                if num_idx < len(numeric):
                    result[param] = numeric[num_idx]
                    num_idx += 1
                else:
                    result[param] = None
        
        return result
    
    def _strip_standard(self, text: str, standard: str) -> str:
        """Remove standard designation from end of text."""
        m = re.search(
            rf'[-\s]*{re.escape(standard)}\s*$', 
            text, re.IGNORECASE
        )
        if m:
            return text[:m.start()].strip()
        for pat in self.STD_PATTERNS:
            m = re.search(rf'[-\s]*{pat}\s*$', text, re.IGNORECASE)
            if m:
                return text[:m.start()].strip()
        return text
    
    @staticmethod
    def _is_numeric(token: str) -> bool:
        """Check if token is numeric (supports comma/dot decimals)."""
        return bool(re.match(r'^\d+(?:[.,]\d+)?$', token))
    
    @staticmethod
    def _normalize(token: str) -> str:
        """Normalize number: comma → dot."""
        return token.replace(',', '.')
