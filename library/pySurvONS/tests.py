from survONS import SurvONS
from sksurv.datasets import load_gbsg2
from utils import get_censored_values, date_discretization
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

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
surv = SurvONS()

indivs = [X[i] for i in range(10)]

# surv.train(x, np.zeros((x.shape[0])), y["time"], y["cens"], diam=1.25)
# print("Concordance index:",surv.score(y["time"], X, y["cens"]))
# surv.iterative_train(x, np.zeros((x.shape[0])), y["time"], y["cens"])

# for i in range(10):
#     print(f"Individuo {i}")
#     print("Predicted time:", surv.predict_time(X[i]))
#     print("Actual time:", y[i]["time"])

surv.plot(indivs, 0, 2500)

dates = ["2024-01-01", "2023-01-01"]
print(date_discretization(dates, depth="year"))