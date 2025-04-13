import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer

import config
from config import Features, TARGET

df = pd.read_csv(config.DATA_PATH)

COLUMNS = df.columns.values
for column in COLUMNS:
    values   = df[column]
    nunique  = values.nunique()
    complete = values.notna().all()
    print(f'{column: <20}{nunique: <10}{complete}')

data = df[Features.ALL + TARGET]

# Drop zero insured value
data = data.query(f'{TARGET[0]} > 0')

transformer = ColumnTransformer(
        [('ohe', OneHotEncoder(sparse_output=False), Features.CATEGORICAL_OHE),
         #('ord', OrdinalEncoder(), Features.CATEGORICAL_ORD),
         ('log', FunctionTransformer(np.log1p), Features.REAL + TARGET)],
    remainder='passthrough')

data = transformer.fit_transform(data)
X, y = data[:, :-1], data[:, -1]

print(X, y)
