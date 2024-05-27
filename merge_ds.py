import sys

import pandas as pd

target, *files = sys.argv[1:]

dfs = []
for name in files:
    dfs.append(pd.read_excel(name))


pd.concat(dfs).drop_duplicates().to_excel(target, index=False)

