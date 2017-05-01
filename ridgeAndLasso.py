#Initialization
#TODO add scaling
#TODO add OneHotEncoder instead of modelmatrix
# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn, Ridge, Lasso, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
import math

# define R equivalent model.matrix function
def modelmatrix(cols,data):
    for col in cols:
        if data[col].dtype.type == np.str_ or data[col].dtype.type == np.object_:
            sufs = data[col].unique()
            for i in sufs:
                data[data[col].name + "_" + i] = 0
                data.loc[data[col] == i,data[col].name + "_" + i] = 1
            data = data.drop([col],axis=1)
    return data

# read file and remove NAs
Hitters = pd.read_csv("Hitters.csv")
Hitters = Hitters.dropna(how="any")

x = modelmatrix(Hitters.drop(['Salary'],axis=1).columns.values,Hitters).astype(np.int32)
y = Hitters['Salary']

np.random.seed(1)
# Split the data set into a training set and a test set.
train = np.random.choice(np.arange(y.size),int(y.size / 2))
test = [i for i in range(y.size) if i not in train]
xtrain = x.iloc[train,]
ytrain = y.iloc[train]
xtest = x.iloc[test,]
ytest = y.iloc[test]

# Fit a linear model using least squares on the training set, and report the test error obtained.
lss = LinearRegression()
lss.fit(xtrain.values,ytrain.values)
print(np.mean((lss.predict(xtest.values)-ytest.values)**2))

# Fit a ridge regression model on the training set, with λλ chosen by cross-validation. Report the test error obtained.
grid = 10 ** (np.arange(-2,5,0.07))
ridgemod = RidgeCV(alphas = grid).fit(xtrain.values,ytrain.values)
print(np.mean((ridgemod.predict(xtest.values)-ytest.values)**2))

# Fit a lasso model on the training set, with λλ chosen by crossvalidation. Report the test error obtained, along with the number of non-zero coefficient estimates.
lassomod = LassoCV(alphas = grid).fit(xtrain.values,ytrain.values)
print(np.mean((lassomod.predict(xtest.values)-ytest.values)**2))
print("Non zero estimates:")
print(lassomod.coef_[lassomod.coef_!=0])


# ridgecoefs = np.array([i.coef_ for i in ridgemod])
# cs = ["blue",
# "green",
# "red",
# "cyan",
# "magenta",
# "yellow"]
#
# for i in range(23):
#     plt.plot(grid,ridgecoefs[:,i],color=cs[i%6])
#     plt.xlabel("Lambda")
#     plt.ylabel("Coefs")
#
# plt.show()

# grid = 10 ** (np.arange(-2,5,0.07))
# grid = grid.tolist()
# ridgemod = np.empty(100,Ridge)
# for i,alp in enumerate(grid):
#     ridgemod[i] = Ridge(alpha = alp).fit(x.values,y.values)
#
# # Fit a lasso model on the training set, with λλ chosen by crossvalidation. Report the test error obtained, along with the number of non-zero coefficient estimates.
#
# ridgecoefs = np.array([i.coef_ for i in ridgemod])
# cs = ["blue",
# "green",
# "red",
# "cyan",
# "magenta",
# "yellow"]
#
# for i in range(23):
#     plt.plot(grid,ridgecoefs[:,i],color=cs[i%6])
#     plt.xlabel("Lambda")
#     plt.ylabel("Coefs")
#
# plt.show()