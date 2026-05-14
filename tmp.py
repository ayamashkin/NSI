# Запустите этот скрипт в той же директории
import sqlite3
import os
import glob

db_path = 'cache/masks.db'

# 1. Закрываем все висячие соединения (VACUUM перестраивает БД целиком)
conn = sqlite3.connect(db_path, isolation_level=None)
cursor = conn.cursor()

# Принудительный checkpoint + truncate WAL
cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
result = cursor.fetchone()
print(f"Checkpoint: busy={result[0]}, log={result[1]}, checkpointed={result[2]}")

# Переключаем в DELETE mode (убираем WAL)
cursor.execute("PRAGMA journal_mode = DELETE;")
print(f"Journal mode: {cursor.fetchone()[0]}")

# Возвращаем в WAL (свежий WAL файл)
cursor.execute("PRAGMA journal_mode = WAL;")
print(f"Journal mode restored: {cursor.fetchone()[0]}")

cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")

conn.close()

# 2. Удаляем старые WAL-файлы если остались
for f in glob.glob('cache/masks.db-*'):
    os.remove(f)
    print(f"Removed stale: {f}")

# 3. Проверяем
conn = sqlite3.connect(db_path)
print(f"Masks: {conn.execute('SELECT count(*) FROM masks').fetchone()[0]}")
conn.close()