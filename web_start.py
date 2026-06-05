#!/usr/bin/env python3
"""
Скрипт запуска web-интерфейса.
Запускает FastAPI backend на порту 8000.
Frontend доступен через Vite dev server на порту 3000 (после npm run dev).
"""

import sys
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
