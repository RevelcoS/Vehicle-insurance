import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer

from os.path import join

path = join('data', 'motor_data14-2018.csv')
df   = pd.read_csv(path)

COLUMNS = df.columns.values
for column in COLUMNS:
    values   = df[column]
    nunique  = values.nunique()
    complete = values.notna().all()
    print(f'{column: <20}{nunique: <10}{complete}')

class Features:
    CATEGORICAL_OHE = ['INSR_TYPE', 'TYPE_VEHICLE', 'USAGE']
    CATEGORICAL_ORD = []
    REAL            = ['PREMIUM']
    ALL             = CATEGORICAL_OHE + CATEGORICAL_ORD + REAL

TARGET = ['INSURED_VALUE']

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
