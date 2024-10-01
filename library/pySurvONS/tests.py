from survONS import SurvONS
from sksurv.datasets import load_gbsg2
from utils import get_censored_values
import numpy as np
import pandas as pd

def cancer_levels(val):
    if val == "I":
        return 1
    elif val == "II":
        return 2
    elif val == "III":
        return 3
    else:
        return 0

x, y = load_gbsg2()

x["horTh"] = x["horTh"].cat.codes
x["menostat"] = x["menostat"].apply(lambda val: 0 if (val == "Pre") else 1)
x["tgrade"] = x["tgrade"].apply(lambda val: cancer_levels(val))

X = x.to_numpy()

a = [1,2,3,4,5,6,1,2]
print(get_censored_values(a, 4))


surv = SurvONS()

indivs = [X[i] for i in range(10)]

surv.train(x, np.zeros((x.shape[0])), y["time"], y["cens"])
surv.plot(X[28], 0, 2500)
surv.plot(indivs, 0, 2500)