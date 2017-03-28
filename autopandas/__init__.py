import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, RobustScaler

from .transformers import CategoryTransformer

class autopandas(BaseEstimator, TransformerMixin):
    def __init__(self, pandas = True, level = 0, ignore=[], drop_at = 0.8, categories = 0.1, **kwargs):

        self.pandas = pandas
        self.level = level

        self.drop_at    = drop_at
        self.categories = categories
        
        self.ignore  = ignore

        self.map = {}
        self.pipeline = []

    def __str__(self, **kwargs):
        if self.map == {}:
            return super().__str__(**kwargs)

        ret = ""
        for column in self.map:
            ret = ret + column + "\t\t" + str(self.map[column][0]) + "\n"
        return(ret)

    def fit(self, X, y=None):
        non_empty = X.count()
        for column in X.columns:
            if column in self.ignore:
                continue
            complete    = ("full" if X.shape[0] == non_empty[column] else "partial")
            unique      = len(X[column].unique())

            # TODO: categorical identification
            if X.dtypes[column] in [np.float64, np.int64]:
                i_type = str(X.dtypes[column])
            elif X.dtypes[column] == np.object:
                # Categorical identification:
                if unique <= non_empty[column] * self.categories:
                    i_type = "categorical"
                else:
                    i_type = np.nan
            else:
                i_type      = np.nan

            self.map[column] = (X.dtypes[column], i_type, complete, unique)
            
# Level 0 - full
        l0 = []
        for key in self.map:
            if not(self.map[key][1] is np.nan) and (self.map[key][2] == 'full'):
                m_str = 'l0_{0}_{1}'.format(self.map[key][1], self.map[key][2])
                method = getattr(self, m_str)
                line   = method(key)
                try:
                    reshape = line[2]
                except:
                    reshape = True
                if reshape:
                    line[1].fit(X[key].reshape(-1,1))
                else:
                    line[1].fit(X[key])
                l0.append(line)
        self.pipeline.append(l0)

# Level 0 - partial
        l05 = []
        for key in self.map:
            if not(self.map[key][1] is np.nan) and (self.map[key][2] == 'partial'):
                m_str = 'l0_{0}_{1}'.format(self.map[key][1], self.map[key][2])
                method = getattr(self, m_str)
                line   = method(key)
                try:
                    reshape = line[2]
                except:
                    reshape = True
                if reshape:
                    line[1].fit(X[key].reshape(-1,1))
                else:
                    line[1].fit(X[key])
                l05.append(line)
        self.pipeline.append(l05)

        return self


    def transform(self, X, y=None):
        res = None
        for pip in self.pipeline:
            for column in pip:
                try:
                    reshape = column[2]
                except:
                    reshape = True
                if reshape:
                    data = column[1].transform(X[column[0]].reshape(-1,1))
                else:
                    data = column[1].transform(X[column[0]])

                if self.pandas:
                    # result will be pandas DataFrame
                    if res is None:
                        res = pd.DataFrame()
                    for i in range(data.shape[1]):
                        if data.shape[1] == 1:
                            column_name = column[0]
                        else:
                            column_name = "{0}_{1}".format(column[0], i)
                        res[column_name] = pd.Series(data[:,i])
                    # column to check if original value is NaN
                    res["{0}_NaN".format(column[0])] = [True if pd.isnull(v) else False for v in X[column[0]]]
                else:
                    # result will be numpy array
                    if res is None:
                        res = data
                    else:
                        res = np.concatenate((res, data), axis=1)

        return res

    def l0_int64_full(self, column):
        return [column, Pipeline([("imputer", Imputer()), ("scale", RobustScaler())])]

    def l0_float64_full(self, column):
        return self.l0_int64_full(column)

    def l0_float64_partial(self, column):
        return self.l0_int64_full(column)

    def l0_int64_partial(self, column):
        return self.l0_int64_full(column)

    def l0_categorical_full(self, column):
        return [column, Pipeline([("categorical", CategoryTransformer(column = column, dropna = False))]), False]

    def l0_categorical_partial(self, column):
        return [column, Pipeline([("categorical", CategoryTransformer(column = column, dropna = True))]), False]
