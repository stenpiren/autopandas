"""
 Collection of transformers for autopandas
"""

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

class DataFrameImputer(TransformerMixin):
    """
    Credits http://stackoverflow.com/a/25562948/1575066
    """

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """

    def fit(self, X, y=None):

        self.fill = pd.Series([
            X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else
            X[c].mean() if X[c].dtype == np.dtype(float) else X[c].median()
            for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill, inplace=False)

class LabelEncoderFix(LabelEncoder):
    """
    Just correction of encoder.
    To be deleted.
    """
    def fit(self, y):
        return super().fit(y.astype(str))

    def transform(self, y):
        return super().transform(y.astype(str))

    def fit_transform(self, y):
        return super().fit_transform(y.astype(str))

class CategoryTransformer(TransformerMixin):
    """Impute missing values.
       Columns of dtype object are imputed with the most frequent value in column.
       Columns of other types are imputed with mean of column.
    """

    def __init__(self, **kwargs):
        self.fill = []
        super().__init__()

    def unique(self, data):
        """ calculation of unique values in array
        numpy calculation fails if contains nan
        """

        unique = []
        counts = []

        for row in data:
            if row in unique:
                counts[unique.index(row)] += 1
            else:
                unique.append(row)
                counts.append(1)

        return unique, counts

    def fit(self, X, y=None):
        """
        Fit transformer
        """
        if y != None:
            raise NotImplementedError

        self.fill = []
        if len(X.shape) == 1 or X.shape[1] == 1:
            unique, counts = self.unique(X)
            ind = np.argmax(counts)
            self.fill.append([ind, unique, counts])
        else:
            columns = X.shape[1]
            for column in range(columns):
                unique, counts = self.unique(X[column])
                ind = np.argmax(counts)
                self.fill.append([ind, unique, counts])

        return self

    def transform(self, X, y=None):
        ret = []

        for rowdata in X:
            resdata = []
            for i in range(len(self.fill[0][1])):
                if rowdata == self.fill[0][1][i]:
                    resdata.append(1)
                else:
                    resdata.append(0)
            ret.append(resdata)

        return np.array(ret)

class LinearImputer(TransformerMixin):
    def __init__(self, **kwargs):
        self.model = None
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        if y is None:
            train = X[~np.isnan(X).any(axis=1)]
        else:
            train = np.concatenate((X, y), axis = 1)[~np.isnan(X).any(axis=1)]

        x_train = train[:, 1:]
        y_train = train[:, 0:1]


        self.model = LinearRegression()
        self.model.fit(x_train, y_train)
        return self

    def transform(self, X, *_):
        result = np.array(X, copy=True)
        for row in result:
            if np.isnan(row[0]):
                row[0] = self.model.predict(row[1:].reshape(1,-1))
        return result
