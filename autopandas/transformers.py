import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

class LabelEncoderFix(LabelEncoder):
    def fit(self, y):
        return super().fit(y.astype(str))

    def transform(self, y):
        return super().transform(y.astype(str))

    def fit_transform(self, y):
        return super().fit_transform(y.astype(str))

class CategoryTransformer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """

    def unique(self, x):
        """ calculation of unique values in array
        numpy calculation fails if contains nan
        """

        unique = []
        counts = []

        for c in x:
            if c[0] in unique:
                counts[unique.index(c[0])] += 1
            else:
                unique.append(c[0])
                counts.append(1)

        return unique, counts

    def fit(self, X, y=None):

        self.fill = []
        if len(X.shape) == 1 or X.shape[1] == 1:
            unique, counts = self.unique(X)
            ind = np.argmax(counts)
            self.fill.append([ind, unique, counts])
        else:
            columns = X.shape[1]
            for c in range(columns):
                unique, counts = self.unique(X[c])
                ind = np.argmax(counts)
                self.fill.append([ind, unique, counts])

        return self

    def transform(self, X, y=None):
        ret = []

        for rowdata in X:
            resdata = []
            for i in range(len(self.fill[0][1])):
                if rowdata[0] == self.fill[0][1][i]:
                    resdata.append(1)
                else:
                    resdata.append(0)
            ret.append(resdata)

        return np.array(ret)

class LinearImputer(TransformerMixin):
    def __init__(self, **kwargs):
        self.model = None
        return super().__init__(**kwargs)

    def fit(self, X, y=None):
        if y is None:
            train = X[~np.isnan(X).any(axis=1)]
        else:
            train  = np.concatenate((X,y), axis = 1)[~np.isnan(data).any(axis=1)]

        x_train = train[:,1:]
        y_train = train[:,0:1]
        

        self.model = LinearRegression()
        self.model.fit(x_train, y_train)
        return self

    def transform(self, X, *_):
        result = np.array(X, copy=True)
        for row in result:
            if np.isnan(row[0]):
                row[0] = self.model.predict(row[1:])
        return result
