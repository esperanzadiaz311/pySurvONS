from survONS import SurvONS
from sksurv.datasets import load_gbsg2
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

surv = SurvONS(x, np.zeros((x.shape[0])), y["time"], y["cens"])

surv.train()
surv.plot(range(10), 0, 2500)