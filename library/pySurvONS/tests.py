import numpy as np
import pandas as pd
import unittest

from lifelines.utils import concordance_index
from survONS import SurvONS
from sksurv.datasets import load_gbsg2
from utils import get_censored_values, date_discretization


def cancer_levels(val):
    if val == "I":
        return 1
    elif val == "II":
        return 2
    elif val == "III":
        return 3
    else:
        return 0

def setup_data():

    x, y = load_gbsg2()

    x["horTh"] = x["horTh"].cat.codes
    x["menostat"] = x["menostat"].apply(lambda val: 0 if (val == "Pre") else 1)
    x["tgrade"] = x["tgrade"].apply(lambda val: cancer_levels(val))

    X = x.to_numpy()

    return x, y, X

class TestSurvONS(unittest.TestCase):

    def setUp(self):
        self.model = SurvONS()
        self.x, self.y, self.X = setup_data()
        self.model.train(self.x, np.zeros((self.x.shape[0])),
                         self.y["time"], self.y["cens"], diam=1.25)

    def test_init(self):
        self.assertIsInstance(self.model, SurvONS)

    def test_train(self):
        expected_beta = [-0.10788718812832382, -0.0199832601998326459908751, 
                         -0.2766667127666671054981923, 0.07727168694191103, 
                         -0.018129000398296426, -0.00190303001903029732701316,
                        -0.2582456229000958, -0.0026548718430288096]
        
        self.assertIsInstance(self.model.beta, np.ndarray)

        for i in range(8):
            self.assertAlmostEqual(expected_beta[i], self.model.beta[i])
        self.assertTrue(self.model.trained)
        self.assertEqual(2.5, self.model.D)

    def test_predict(self):

        not_trained = SurvONS()
        self.assertIsNone(not_trained.predict(self.X[25], 10))

        expected_list = [0.9992604640901819, 0.998106803691939, 0.9999707980905742]
        expected_val = 0.9868755846079249

        preds = self.model.predict(self.X[:3], 10)
        for i in range(3):
            self.assertAlmostEqual(preds[i], expected_list[i])

        self.assertAlmostEqual(self.model.predict(self.X[25], 10), expected_val)
        self.assertIsNone(self.model.predict([], 10))
        
    def test_predict_time(self):

        not_trained = SurvONS()
        self.assertIsNone(not_trained.predict_time(self.X[25]))

        expected_list = [1285.9065008815317, 1218.2353727805432, 1327.7781349144761]
        expected_val = 674.8058056995441

        preds = self.model.predict_time(self.X[:3])
        for i in range(3):
            self.assertAlmostEqual(preds[i], expected_list[i])

        self.assertAlmostEqual(self.model.predict_time(self.X[25]), expected_val)
        self.assertIsNone(self.model.predict_time([]))

    def test_score(self):
        not_trained = SurvONS()
        self.assertIsNone(not_trained.score(self.y["time"], self.X,
                                             self.y["cens"]))
        
        self.assertIsNone(self.model.score([], self.X, self.y["cens"]))
        self.assertIsNone(self.model.score(self.y["time"], [], self.y["cens"]))
        self.assertIsNone(self.model.score(self.y["time"], self.X, []))

        self.assertIsNone(self.model.score(self.y["time"][:2], self.X,
                                             self.y["cens"]))
        self.assertIsNone(self.model.score(self.y["time"], self.X[:2],
                                             self.y["cens"]))
        self.assertIsNone(self.model.score(self.y["time"], self.X,
                                             self.y["cens"][:2]))

        expected_val = 0.5715251893711675
        self.assertAlmostEqual(expected_val, 
                               self.model.score(self.y["time"], self.X,
                                             self.y["cens"]))

    def test_iterative_train(self):

        not_trained = SurvONS()
        self.assertFalse(not_trained.iterative_train(self.x, np.zeros((self.x.shape[0])),
                                   self.y["time"], self.y["cens"]))

        expected_beta = [-0.0797287831, -0.0222892695, -0.1395073547,
                         0.3585469799, -0.0018600354, -0.0275434332,
                         -0.2333098762, -0.0281262133]

        self.model.iterative_train(self.x, np.zeros((self.x.shape[0])),
                                   self.y["time"], self.y["cens"])

        self.assertIsInstance(self.model.beta, np.ndarray)

        for i in range(8):
            self.assertAlmostEqual(expected_beta[i], self.model.beta[i])
        self.assertTrue(self.model.trained)
        self.assertEqual(1.8, self.model.D)
        
if __name__ == '__main__':

    unittest.main()
