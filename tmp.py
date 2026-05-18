# 1. Есть ли ключ?
cat secrets/mts_ai_key.txt

# 2. Создается ли клиент при старте?
python -c "
from config.settings import get_settings
settings = get_settings()
print('mask_generation.default_service:', settings.mask_generation.default_service)
print('api.mts_ai.base_url:', settings.api.get('mts_ai', {}).base_url if 'mts_ai' in settings.api else 'NOT FOUND')
print('api.mts_ai.api_key present:', bool(settings.api.get('mts_ai', {}).api_key) if 'mts_ai' in settings.api else False)
"