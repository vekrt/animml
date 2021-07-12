import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t
from scipy.stats import f
from scipy.special import hyp2f1
from sklearn.linear_model import LinearRegression, Lasso, LassoLars

data = pd.read_csv("prostate.dat", sep="\t")

idx = data["train"] == "T"

data = data[idx]

cols = data.columns[1:-1]
data = data[cols]

#data.to_csv("clean_data.dat", index=False)

X = data[cols[:-1]]
X_p = X
X = sm.add_constant(X)


Y = data[cols[-1]]
model = sm.OLS(Y,X)
results = model.fit()
model = sm.OLS(Y,X)
results_reg = model.fit_regularized(alpha=0.1, L1_wt=0.0)
print (results.summary())
print (results_reg.params)
print (np.linalg.inv(X.T @ X + X.shape[0] * 0.1 * np.eye(X.shape[1])) @ X.T @ Y)
print ("Norm_stat: ", np.sum(np.abs(results_reg.params)))

"""
nu = 20.0
x = 0.1
print (t.cdf(x, nu))
print (hyp2f1(0.5, (nu + 1.0)/2.0, 1.5, -x*x/nu))
print (t.ppf(0.025, 50.0))
print (1.0 - f.cdf(16.47, 8, 58))
print (f.cdf(1.1, 9.0, 2.0))

reg = LinearRegression().fit(X_p, Y)

print (reg.intercept_)
print (reg.coef_)
#"""
"""
clf = Lasso(alpha=0.1).fit(X_p, Y)

print (clf.intercept_)
print (clf.coef_)
print ("Norm: ", np.abs(clf.intercept_) + np.sum(np.abs(clf.coef_)))
#"""

xx = X.T @ X
print (xx)
print (np.linalg.det(xx))
print (np.linalg.eig(np.cov(X_p.T)))
