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

#a = [1,2,3,4,5,6,1,2]
# print(get_censored_values(a, 4))


surv = SurvONS()

indivs = [X[i] for i in range(10)]

surv.train(x, np.zeros((x.shape[0])), y["time"], y["cens"])
print("Concordance index:",surv.score(y["time"], X, y["cens"]))

for i in range(10):
    print(f"Individuo {i}")
    print("Predicted time:", surv.predict_time(X[i]))
    print("Actual time:", y[i]["time"])

# surv.plot(X[28], 0, 2500)
# surv.plot(indivs, 0, 2500)

# names = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
# events = [1, 2, 3, 4, 5]
# preds = [3, 1, 2, 5, 4]
# df = pd.DataFrame(data={'Churn times': events, 'Predictions': preds}, index=names)
# print(df)
# print(f'Concordance index: {concordance_index(events, preds)}')

# dates = np.array(['2023-09-27 08:45', '2023-01-27 08:46', 
#                   '2023-09-27 08:55', '2023-09-27 09:45'])
# discretized = date_discretization(dates, "month")
# print(discretized)