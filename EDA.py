import pandas as pd
import config

df = pd.read_csv(config.DATA_PATH)

NAMES = ['', 'NUNIQUE', 'NOTNA']
print('{0:<20}{1:<10}{2}'.format(*NAMES))

COLUMNS = df.columns.values
for column in COLUMNS:
    values   = df[column]
    nunique  = values.nunique()
    complete = values.notna().all()
    print(f'{column: <20}{nunique: <10}{complete}')

