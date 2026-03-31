import pandas as pd
import re

# Чтение файла
file_id = "d231899b-a83e-4ce6-85ba-7c59bd6780d0"
df = pd.read_excel(file_id)

# Приведение имён колонок к удобному виду
cols = [str(c).strip().lower() for c in df.columns]
df.columns = cols

# Поиск колонки с наименованием
name_col = None
for c in cols:
   if 'наимен' in c:
       name_col = c
       break
if name_col is None:
   name_col = cols[1]  # fallback на вторую колонку

# Функция категоризации с учётом новой категории "Прокат"
def categorize(name: str) -> str:
   n = str(name).lower()
   # Крепёж
   if re.search(r'винт|болт|гайк|шайб|шпильк|анкер|саморез|заклепк', n):
       return 'Метизы'
   # Трубы и профиль
   if re.search(r'труб|швеллер|уголок|балк|профиль|труба', n):
       return 'Трубы и профиль'
   # Лист/плита/рулон
   if re.search(r'лист|плита|рулон', n):
       return 'Листовые материалы'
   # Круг/квадрат/шестигранник
   if re.search(r'круг|квадрат|шестигранник', n):
       return 'Сортовой прокат'
   # Проволока/сетка
   if re.search(r'проволок|сетк', n):
       return 'Проволока и сетка'
   # Лента/полоса
   if re.search(r'лент|полос', n):
       return 'Лента и полоса'
   # Заготовки/отливки/поковки
   if re.search(r'заготовк|отливк|поковк', n):
       return 'Заготовки и полуфабрикаты'
   # Изоляция/уплотнение
   if re.search(r'изол|уплотн', n):
       return 'Изоляция и уплотнение'
   # Химия/лакокрасочные материалы
   if re.search(r'лак|краск|грунтов|эмаль|смол|клей|гермет', n):
       return 'Химия и ЛКМ'
   # Электрика/автоматика
   if re.search(r'кабел|провод|автомат|реле|контактор', n):
       return 'Электрика и автоматика'
   # Инструмент/оснастка
   if re.search(r'инструмент|оснастк', n):
       return 'Инструмент и оснастка'
   # Строительные материалы
   if re.search(r'цемент|бетон|кирпич|раствор', n):
       return 'Строительные материалы'
   # Прокат (сталь, конструкционные, калиброванные, нержавеющие и пр.)
   if re.search(r'^ст\.сорт\.нерж\.|ст\.констр\.калибр\.|ст\.сорт\.|ст\.констр\.', n):
       return 'Прокат'
   # Прочие материалы (металл, сталь, чугун, алюминий)
   if re.search(r'сталь|чугун|алюмин|металл', n):
       return 'Материалы'
   # Прочее оборудование и комплектующие
   if re.search(r'подшипник|ступиц|редуктор|насос', n):
       return 'Оборудование и комплектующие'
   # Прочее (по умолчанию)
   return 'Прочее'

df['Категория'] = df[name_col].astype(str).apply(categorize)
category_counts = df['Категория'].value_counts(dropna=False)
category_examples = df.groupby('Category')[name_col].agg(lambda x: x.dropna().unique()[:3].tolist())

top_n = 10
result = []
for cat, count in category_counts.head(top_n).items():
   ex = category_examples.get(cat, [])
   result.append({'Категория': cat, 'Количество': int(count), 'Примеры': ex})
print(result)