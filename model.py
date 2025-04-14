import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression

import config
from config import Features, TARGET

def get_data(path=config.DATA_PATH):
    df = pd.read_csv(path)
    data = df[Features.ALL + TARGET]

    # Drop zero insured value
    data = data.query(f'{TARGET[0]} > 0')

    transformer = ColumnTransformer(
            [('ohe', OneHotEncoder(sparse_output=False,
                                   handle_unknown='ignore'),
              Features.CATEGORICAL_OHE),
             #('imp', SimpleImputer(missing_values=pd.NA), Features.REAL),
             ('log', FunctionTransformer(np.log1p), Features.REAL + TARGET)],
        remainder='passthrough')

    # Drop nans
    data = data[~pd.isna(data).any(axis=1)]

    data = transformer.fit_transform(data)
    X, y = data[:, :-1], data[:, -1]

    return X, y


if __name__ == '__main__':
    X, y = get_data()

    X = sm.add_constant(X, prepend=False)
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.20, random_state=config.RANDOM_SEED)

    model   = sm.OLS(y_train, X_train)
    results = model.fit()
    print(results.summary())
