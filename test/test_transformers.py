import unittest
import numpy as np
from autopandas.transformers import LinearImputer

class Test_transformers(unittest.TestCase):
    def test_LinearImputter(self):
        data = np.array([[1,1],[2,2], [np.nan, 3]])
        i = LinearImputer()
        result = i.fit_transform(data)
        print(result)
        self.assertTrue(abs(result[2,0] - 3) < 0.00001 )

        data = np.array([[2,1],[4,2], [6,3], [np.nan, 4]])
        i = LinearImputer()
        result = i.fit_transform(data)
        self.assertTrue(abs(result[3,0] - 8) < 0.00001)

        data = np.array([[2,1,1],[3,1,2], [4,2,2], [np.nan, 3, 2]])
        i = LinearImputer()
        result = i.fit_transform(data)
        self.assertTrue(abs(result[3,0] - 5) < 0.00001)


if __name__ == '__main__':
    unittest.main()
