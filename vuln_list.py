import sys

import pandas as pd

source, target = sys.argv[1:3]

data = pd.read_excel(source, skiprows=2)

cols = data.columns
name_col = cols[4]
type_col = cols[6]

# types:
# ПО виртуализации/ПО виртуального программно-аппаратного средства
# Сетевое программное средство
# Средство защиты
# Программное средство АСУ ТП
# ПО программно-аппаратных средств защиты
# Сетевое средство
# ПО программно-аппаратного средства АСУ ТП
# Прикладное ПО информационных систем
# Микропрограммный код
# ПО программно-аппаратного средства
# СУБД
# Микропрограммный код аппаратных компонент компьютера
# Операционная система
# ПО сетевого программно-аппаратного средства
# Программное средство защиты
# Средство АСУ ТП

pt_types = {
    'Средство защиты',
    'ПО программно-аппаратных средств защиты',
    'Программное средство защиты',
}

clean = set()

untyped = frozenset()
for i, row in data.iterrows():
    name_value = row[name_col]
    if name_value != name_value:
        continue
    names = {s.strip() for s in name_value.split(',')}
    names = list(filter(None, names))

    type_value = row[type_col]
    if type_value != type_value:  # untyped
        types = untyped
    else:
        types = {s.strip() for s in type_value.split(',')}

    pt = bool(types & pt_types)

    yes, no = 0, 1
    if pt:
        yes, no = 1, 0

    clean.update((name, yes, no) for name in names)

print(f'Clean: {len(clean)}')

df = pd.DataFrame(clean, columns=['name', 'yes', 'no'])
df.to_excel(target, index=False)
