import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.linear_model as sklm
import sklearn.metrics as skm
import sklearn.ensemble as ske
import sklearn.neural_network as sknn
import sklearn.pipeline as skp
import sklearn.tree as skt
import sklearn.preprocessing as skpp
import matplotlib.pyplot as plt
from typing import List
from math import sqrt


import os
for dirname, _, filenames in os.walk('./Dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Project Objective: Predict raises or lowerings in groundwater levels
# 
# **The expected results are as follows:**
# 1. Generate predictions for specific areas based on current water levels and time of year.
# 2. Provide leads for strategic public water supply programs.

# ## Data Pre-Process Phase
# **Clean and Prepare data for regression model consumption**
# 
# 1. Set measurement as data frame index.
# 2. Drop rows that contain null values.
# 3. Select a sample measurement statation.
# 4. Create a column with shifted values (results) to compare predictions. This column will help to compare expected vs real values.

# In[ ]:


def init_daily_data(station: str="09N03E08C003M") -> pd.DataFrame:
    daily_data_by_sample_station: pd.DataFrame = pd.read_csv('./Dataset/gwl-daily.csv')
    daily_data_by_sample_station["MSMT_DATE"] = pd.to_datetime(daily_data_by_sample_station["MSMT_DATE"])
    daily_data_by_sample_station = daily_data_by_sample_station.loc[
        daily_data_by_sample_station["WSE"].notnull(), 
        ["STATION", "MSMT_DATE", "WLM_RPE", "WLM_RPE_QC", "WLM_GSE", "WLM_GSE_QC", "RPE_WSE", "RPE_WSE_QC", "GSE_WSE", "GSE_WSE_QC", "WSE", "WSE_QC"]]
    daily_data_by_sample_station = daily_data_by_sample_station.loc[
        daily_data_by_sample_station["WLM_GSE"].notnull(), 
        ["STATION", "MSMT_DATE", "WLM_RPE", "WLM_RPE_QC", "WLM_GSE", "WLM_GSE_QC", "RPE_WSE", "RPE_WSE_QC", "GSE_WSE", "GSE_WSE_QC", "WSE", "WSE_QC"]]
    
    # Select Sample Station
    daily_data_by_sample_station = daily_data_by_sample_station[daily_data_by_sample_station["STATION"] == station]
    daily_data_by_sample_station = daily_data_by_sample_station.set_index("MSMT_DATE").sort_index()
    # target = daily_data_by_sample_station["GSE_WSE"]
    daily_data_by_sample_station["target"] = daily_data_by_sample_station.shift(-1)["GSE_WSE"]  # Create a column with shifted values (results) to compare predictions
    daily_data_by_sample_station = daily_data_by_sample_station.iloc[:-1, :].copy()

    return daily_data_by_sample_station

daily_data_by_sample_station = init_daily_data()


# In[ ]:


predictors: List[str] = ["WLM_RPE", "WLM_GSE", "RPE_WSE", "GSE_WSE", "WSE"]
models_without_coef: tuple = (skp.Pipeline, ske._forest.RandomForestRegressor, sklm._ransac.RANSACRegressor, ske._bagging.BaggingRegressor, 
                              sknn._multilayer_perceptron.MLPRegressor, skt._classes.DecisionTreeRegressor, ske._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor,
                             ske._gb.GradientBoostingRegressor, ske._voting.VotingRegressor)


# ### Create Linear Regression Function
# 1. Create training and testing data frames.
# 2. Fit the model using a predictors set (X value) and target measurement training set (y value).
# 3. Create a new data frame with the predictions and expected (real) values.
# 4. Print and plot Model's performance.

# In[ ]:


def predict_linear_regressors(dataframe: pd.DataFrame, model: sklm, predictors: List[str]=["WLM_RPE", "WLM_GSE", "RPE_WSE", "GSE_WSE", "WSE"]) -> pd.DataFrame:
    dataframes: List[pd.DataFrame] = dataframe_training_testing_subsets(dataframe)
    train: pd.DataFrame = dataframes[0]
    test: pd.DataFrame = dataframes[1]
        
    model.fit(train[predictors], train["target"])
    predictions : [] = model.predict(test[predictors]) # list of predictions 


    
    combined: pd.DataFrame = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    
    if not (isinstance(model, models_without_coef)):
        feature_statistics(model)
        print("Independent term: %.3f" %model.intercept_)
    
    model_metrics(test["target"], predictions)   
    combined.plot(figsize=(25,10), fontsize=15)
    
    return combined


# ### Create Training and Testing Data frames Function
# 1. Based on a given data frame, create separate training and test data frame.
# 2. Each subsequent data frame can have a start date given as function parameter.
# 3. The function returns a list of data frames (Training, Test).

# In[ ]:


def dataframe_training_testing_subsets(dataframe: pd.DataFrame, train_date: str="2021-12-31", test_date: str="2022-1-1") -> List[pd.DataFrame]:
    res: List[pd.DataFrame] = []
    train: pd.DataFrame = dataframe.loc[:train_date]
    test: pd.DataFrame = dataframe.sort_index().loc[test_date:]
    res.extend([train, test])
    
    return res


# ### Feature statistics model 
# 1. For each model's feature, print its coefficient statistic.

# In[ ]:


def feature_statistics(model: sklm) -> None:
    items = []
    coef: np.ndarray = model.coef_
    for index, item in enumerate(model.feature_names_in_):
        print(f"{item} - Coeficcient: {coef[index]}")
    print("\n")


# ### Model Metrics
# 1. Print model performance statistics

# In[ ]:


def model_metrics(target: pd.DataFrame, predictions: []) -> None:
    print("Explained variance regression score: %.3f" %skm.explained_variance_score(target, predictions))
    print("Maximum residual error: %.3f" %skm.max_error(target, predictions)) # rediual error is the difference between the observed value and the predicted value
    print("Mean squared error: %.3f" %skm.mean_squared_error(target, predictions))
    print("Coeficcient of determination: %.3f" %skm.r2_score(target, predictions))
    print("Root-mean-square deviation: %.3f" %sqrt(skm.mean_squared_error(target, predictions)))


# # Classical Linear Regressors
# 1. Linear Regresion (Ordinary least squares Linear Regression).
# 2. Ridge (Linear least squares with I2 regularization.
# 3. Ridge CV (Ridge regression with built-in cross-validation).
# 4. SGD Regressor (Linear rmodel fitted by minimizing a regularized empirial loss with SGD).

# ## Linear Regression
# **Ordinary least squares Linear Regression.**
# 
# LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

# In[ ]:


linear_reg = sklm.LinearRegression()
predict_linear_regressors(daily_data_by_sample_station, linear_reg)


# ## Ridge 
# **Linear least squares with l2 regularization.**
# 
# Minimizes the objective function:
# > ||y - Xw||^2_2 + alpha * ||w||^2_2
# 
# This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization. This estimator has built-in support for multi-variate regression (i.e., when y is a 2d-array of shape (n_samples, n_targets)).

# In[ ]:


ridge_reg = sklm.Ridge(alpha=1.0)
predict_linear_regressors(daily_data_by_sample_station, ridge_reg)


# ## Ridge CV
# **Ridge regression with built-in cross-validation.**
# By default, it performs efficient Leave-One-Out Cross-Validation.

# In[ ]:


ridge_cv_reg = sklm.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
predict_linear_regressors(daily_data_by_sample_station, ridge_cv_reg)


# ## SGD Regressor
# **Linear model fitted by minimizing a regularized empirical loss with SGD.**
# 
# SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate).
# 
# The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection.
# 
# This implementation works with data represented as dense numpy arrays of floating point values for the features.

# In[ ]:


import sklearn.pipeline as skp
from sklearn.preprocessing import StandardScaler

dataframe = dataframe_training_testing_subsets(daily_data_by_sample_station)

sgd = sklm.SGDRegressor(max_iter=1000, tol=1e-3)
sgd_reg = skp.make_pipeline(StandardScaler(),sgd)
predict_linear_regressors(daily_data_by_sample_station, sgd_reg)


# # Regressors with Variable Selection
# The following estimators have built-in variable selection fitting procedures, but any estimator using a L1 or elastic-net penalty also performs variable selection: typically SGDRegressor or SGDClassifier with an appropriate penalty.

# ## Elastic Net Regressor
# Linear regression with combined L1 and L2 priors as regularizer.
# 
# Minimizes the objective function:
# 
# > 1 / (2 * n_samples) * ||y - Xw||^2_2
# > + alpha * l1_ratio * ||w||_1
# > + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
# 
# If you are interested in controlling the L1 and L2 penalty separately, keep in mind that this is equivalent to:
# 
# > a * ||w||_1 + 0.5 * b * ||w||_2^2
# 
# where:
# 
# > alpha = a + b and l1_ratio = a / (a + b)
# 
# The parameter l1_ratio corresponds to alpha in the glmnet R package while alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable, unless you supply your own sequence of alpha.

# In[ ]:


elastic_net_reg = sklm.ElasticNet(random_state=0)

predict_linear_regressors(daily_data_by_sample_station, elastic_net_reg)


# ## Elastic Net CV
# Elastic Net model with iterative fitting along a regularization path.

# In[ ]:


elastic_net_cv_reg = sklm.ElasticNetCV(cv=5, random_state=0)

predict_linear_regressors(daily_data_by_sample_station, elastic_net_cv_reg)


# ## Lars Regression
# Least Angle Regression model a.k.a. LAR.

# In[ ]:


# from sklearn.pipeline import make_pipeline

# lars_reg = make_pipeline(StandardScaler(with_mean=False), sklm.Lars(n_nonzero_coefs=1))
lars_reg = sklm.Lars(n_nonzero_coefs=1)
predict_linear_regressors(daily_data_by_sample_station, lars_reg)


# ## Lars CV 
# Cross-validated Least Angle Regression model.

# In[ ]:


lars_cv_reg = sklm.LarsCV(cv=5)
predict_linear_regressors(daily_data_by_sample_station, lars_cv_reg)


# ## Lasso regressor
# Linear Model trained with L1 prior as regularizer (aka the Lasso).
# 
# The optimization objective for Lasso is:
# 
# > (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
# 
# Technically the Lasso model is optimizing the same objective function as the Elastic Net with l1_ratio=1.0 (no L2 penalty).

# In[ ]:


lasso_reg = sklm.Lasso(alpha=0.1)

predict_linear_regressors(daily_data_by_sample_station, lasso_reg)


# ## Lasso CV regresor
# Lasso linear model with iterative fitting along a regularization path.
# 
# The best model is selected by cross-validation.
# 
# The optimization objective for Lasso is:
# 
# > (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

# In[ ]:


lasso_cv_reg = sklm.LassoCV(cv=5, random_state=0)

predict_linear_regressors(daily_data_by_sample_station, lasso_cv_reg)


# ## Lasso Lars
# Lasso model fit with Least Angle Regression a.k.a. Lars.
# 
# It is a Linear Model trained with an L1 prior as regularizer.
# 
# The optimization objective for Lasso is:
# 
# > (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

# In[ ]:


lasso_lars_reg = sklm.LassoLars(alpha=0.01)

predict_linear_regressors(daily_data_by_sample_station, lasso_lars_reg)


# ## Lasso Lars CV
# Cross-validated Lasso, using the LARS algorithm.
# 
# The optimization objective for Lasso is:
# 
# > (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

# In[ ]:


lasso_lars_cv_reg = sklm.LassoLarsCV(cv=5)

predict_linear_regressors(daily_data_by_sample_station, lasso_lars_cv_reg)


# ## Lasso Larts IC 
# Lasso model fit with Lars using BIC or AIC for model selection.
# 
# The optimization objective for Lasso is:
# 
# > (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
# 
# AIC is the Akaike information criterion [2] and BIC is the Bayes Information criterion [3]. Such criteria are useful to select the value of the regularization parameter by making a trade-off between the goodness of fit and the complexity of the model. A good model should explain well the data while being simple.

# In[ ]:


lasso_lars_criterion_reg = sklm.LassoLarsIC(criterion='bic')

predict_linear_regressors(daily_data_by_sample_station, lasso_lars_criterion_reg)


# ## Orthogonal Matching Pursuit regressor
# Orthogonal Matching Pursuit model (OMP).

# In[ ]:


orthogonal_matching_pursuit_reg = sklm.OrthogonalMatchingPursuit()

predict_linear_regressors(daily_data_by_sample_station, orthogonal_matching_pursuit_reg)


# ## Orthogonal Matching Pursuit CV
# Cross-validated Orthogonal Matching Pursuit model (OMP).

# In[ ]:


orthogonal_matching_pursuit_cv_reg = sklm.OrthogonalMatchingPursuitCV()

predict_linear_regressors(daily_data_by_sample_station, orthogonal_matching_pursuit_cv_reg)


# # Bayesian regresors
# ## Bayessian ARD regressor
# Bayesian ARD regression.
# 
# Fit the weights of a regression model, using an ARD prior. The weights of the regression model are assumed to be in Gaussian distributions. Also estimate the parameters lambda (precisions of the distributions of the weights) and alpha (precision of the distribution of the noise). The estimation is done by an iterative procedures (Evidence Maximization)

# In[ ]:


ard_reg = sklm.ARDRegression()

predict_linear_regressors(daily_data_by_sample_station, ard_reg)


# ## Bayesian Ridge regression
# Fit a Bayesian ridge model. See the Notes section for details on this implementation and the optimization of the regularization parameters lambda (precision of the weights) and alpha (precision of the noise).

# In[ ]:


bayessian_ridge_reg = sklm.BayesianRidge()

predict_linear_regressors(daily_data_by_sample_station, bayessian_ridge_reg)


# # Outlier-Robust Regressors
# ## Huber regressor
# L2-regularized linear regression model that is robust to outliers.
# 
# The Huber Regressor optimizes the squared loss for the samples where |(y - Xw - c) / sigma| < epsilon and the absolute loss for the samples where |(y - Xw - c) / sigma| > epsilon, where the model coefficients w, the intercept c and the scale sigma are parameters to be optimized. The parameter sigma makes sure that if y is scaled up or down by a certain factor, one does not need to rescale epsilon to achieve the same robustness. Note that this does not take into account the fact that the different features of X may be of different scales.
# 
# The Huber loss function has the advantage of not being heavily influenced by the outliers while not completely ignoring their effect.

# In[ ]:


huber_reg = sklm.HuberRegressor()

predict_linear_regressors(daily_data_by_sample_station, huber_reg)


# ## Quantile regressor
# Linear regression model that predicts conditional quantiles.
# 
# The linear QuantileRegressor optimizes the pinball loss for a desired quantile and is robust to outliers.
# 
# This model uses an L1 regularization like Lasso.

# In[ ]:


from sklearn.utils.fixes import sp_version, parse_version
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

quantile_reg = sklm.QuantileRegressor(quantile=0.8, solver=solver)

predict_linear_regressors(daily_data_by_sample_station, quantile_reg)


# ## Theil-Sen regressor
# Theil-Sen Estimator: robust multivariate regression model.
# 
# The algorithm calculates least square solutions on subsets with size n_subsamples of the samples in X. Any value of n_subsamples between the number of features and samples leads to an estimator with a compromise between robustness and efficiency. Since the number of least square solutions is “n_samples choose n_subsamples”, it can be extremely large and can therefore be limited with max_subpopulation. If this limit is reached, the subsets are chosen randomly. In a final step, the spatial median (or L1 median) is calculated of all least square solutions.

# In[ ]:


theil_sen_reg = sklm.TheilSenRegressor(random_state=0)

predict_linear_regressors(daily_data_by_sample_station, theil_sen_reg)


# ## RanSac regressor
# RANSAC (RANdom SAmple Consensus) algorithm.
# RANSAC is an iterative algorithm for the robust estimation of parameters from a subset of inliers from the complete data set.

# In[ ]:


ransac_reg = sklm.RANSACRegressor(random_state=0)

predict_linear_regressors(daily_data_by_sample_station, ransac_reg)


# # Generalized Linear Models (GLM) for regression
# ## Poisson regressor
# Generalized Linear Model with a Poisson distribution.
# 
# This regressor uses the ‘log’ link function.

# In[ ]:


poisson_reg = sklm.PoissonRegressor()

predict_linear_regressors(daily_data_by_sample_station, poisson_reg)


# ## Tweedie regressor
# Generalized Linear Model with a Tweedie distribution.
# 
# This estimator can be used to model different GLMs depending on the power parameter, which determines the underlying distribution.

# In[ ]:


tweedie_reg = sklm.TweedieRegressor()

predict_linear_regressors(daily_data_by_sample_station, tweedie_reg)


# ## Gamma regressor
# Generalized Linear Model with a Gamma distribution.
# 
# This regressor uses the ‘log’ link function.

# In[ ]:


gamma_reg = sklm.GammaRegressor()

predict_linear_regressors(daily_data_by_sample_station, gamma_reg)


# # Miscellaneous
# ## Passive Agressive regressor

# In[ ]:


passive_aggressive_reg = sklm.PassiveAggressiveRegressor(max_iter=100, random_state=0, tol=1e-3)

predict_linear_regressors(daily_data_by_sample_station, passive_aggressive_reg)


# ## ARIMA
# Autoregressive Integrated Moving Average model


dataframes: List[pd.DataFrame] = dataframe_training_testing_subsets(daily_data_by_sample_station)
train: pd.DataFrame = dataframes[0]
test: pd.DataFrame = dataframes[1]

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(train["GSE_WSE"][:], label='Original Series')
axes[0].plot(train["GSE_WSE"][:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)


# Seasinal Dei
axes[1].plot(train["GSE_WSE"][:], label='Original Series')
axes[1].plot(train["GSE_WSE"][:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Depth below ground surface or the Distance from the ground surface to the water surface in feet ', fontsize=16)
plt.show()


# In[ ]:


import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(train["GSE_WSE"], start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=6,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()


# daily_data_pre["GSE_WSE"].plot(figsize=(12,5))

# In[ ]:


# Forecast
n_periods = 24
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(test["GSE_WSE"].index[-1], periods = n_periods, freq='MS')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(test["GSE_WSE"])
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("SARIMA - Final Forecast of Ground Water Levels")
plt.show()


# In[ ]:


from statsmodels.tsa.stattools import adfuller

def ad_test(dataset: pd.DataFrame)-> None:
    dftest = adfuller(dataset, autolag="AIC")
    print("1. ADF: ", dftest[0])
    print("2. P-Value: ", dftest[1])
    print("3. Num of Lags: ", dftest[2])
    print("4. Num of Observations Used for ADF Regression and Critical Values Calculation: ", dftest[3])
    print("5. Critical Values: ")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)


# In[ ]:


dataframes: List[pd.DataFrame] = dataframe_training_testing_subsets(daily_data_by_sample_station)
train: pd.DataFrame = dataframes[0]
test: pd.DataFrame = dataframes[1]
    
ad_test(train["GSE_WSE"])
# ad_test(train[predictors])


# In[27]:


from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")
stepwise_fit = auto_arima(train["GSE_WSE"], trace=True, suppress_warnings=True)
stepwise_fit.summary()


# In[ ]:


model_auto = auto_arima(train["GSE_WSE"])
model_auto


# In[ ]:


model_auto.summary()


# In[ ]:


from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
model_auto = auto_arima(train["GSE_WSE"], exogenous=[["WLM_RPE", "WLM_GSE", "RPE_WSE", "WSE"]], m=7,
                       max_order=None, max_p=9, max_q=9, max_d=2, max_P=4, max_Q=4, max_D=2,
                       maxiter=25, alpha=0.05, n_jobs=-1, trend="ct", information_criterion="oob",
                       out_of_sample_size=int(len(train)*0.2), trace=True, suppress_warnings=True)
model_auto.summary()
# aRima to predict the groundwater levels

# In[ ]:


from statsmodels.tsa.arima.model import ARIMA

dataframes: List[pd.DataFrame] = dataframe_training_testing_subsets(daily_data_by_sample_station)
train: pd.DataFrame = dataframes[0]
test: pd.DataFrame = dataframes[1]

# model = ARIMA(train["GSE_WSE"], order=(3,1,4))
model = ARIMA(train["GSE_WSE"], order=(2,0,3))
model_fit = model.fit()
model_fit.summary()


# In[ ]:


from statsmodels.graphics.tsaplots import plot_predict

plot_predict(model_fit, dynamic=False)


# In[ ]:


from statsmodels.tsa.arima.model import ARIMA
# Arima is a class that represents an ARIMA model. It is instantiated with the 
# training data and the order of the model.
# The order is a tuple that describes the (p,d,q) order of the model. p is the number of autoregressive terms,
#  d is the number of nonseasonal differences needed for stationarity, 
# and q is the number of lagged forecast errors in the prediction equation.
model = ARIMA(train["GSE_WSE"], order=(1,1,2))
model = model.fit()
model.summary()


# In[ ]:


start = len(train)
end = len(train) + len(test)-1
predictions = model.predict(start=start, end=end, typ="levels")
predictions.index = daily_data_by_sample_station.index[start:end+1]
predictions


# In[ ]:


predictions.plot(legend=True)
test["GSE_WSE"].plot(legend=True)


# In[ ]:


from math import sqrt

# test["GSE_WSE"].mean()
rmse = sqrt(skm.mean_squared_error(predictions, test["GSE_WSE"]))
rmse


# In[ ]:


predictions.index = test.index
combined: pd.DataFrame = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "prediction"]
combined["difference"] = combined["prediction"] - combined["actual"]
combined
# plt.scatter(combined["actual"], combined["prediction"])
plt.plot(combined["actual"], label="Actual")
plt.plot(combined["prediction"], label="Predicted")
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
plt.title("ARIMA Prediction Results")
plt.show()


# # Neural Networks
# ## Multi-layer Perceptron regressor.
# Multi-layer Perceptron regressor.
# 
# This model optimizes the squared error using LBFGS or stochastic gradient descent.

# In[ ]:


mlp_reg = sknn.MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=100, random_state=42)

predict_linear_regressors(daily_data_by_sample_station, mlp_reg)


# ## Bernoulli Restricted Boltzmann Machine (RBM).
# A Restricted Boltzmann Machine with binary visible units and binary hidden units. Parameters are estimated using Stochastic Maximum Likelihood (SML), also known as Persistent Contrastive Divergence (PCD) [2].
# 
# The time complexity of this implementation is O(d ** 2) assuming d ~ n_features ~ n_components.

# In[ ]:


# bernoulli_rbm_reg = sknn.BernoulliRBM(n_components=2)

# predict_linear_regressors(daily_data_by_sample_station, bernoulli_rbm_reg)


# # Ensemble Methods
# ## Bagging regressor
# A Bagging regressor is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.
# 
# This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [1]. If samples are drawn with replacement, then the method is known as Bagging [2]. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [3]. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [4].

# In[ ]:


bagging_reg = ske.BaggingRegressor(n_estimators=10, random_state=0)

predict_linear_regressors(daily_data_by_sample_station, bagging_reg)


# ## Random forest regressor.
# 
# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

# In[ ]:


random_forest_reg = ske.RandomForestRegressor(max_depth=2, random_state=0)
predict_linear_regressors(daily_data_by_sample_station, random_forest_reg)


# # Tree-based Models
# ## Decision Tree regressor

# In[ ]:


decision_tree_reg = skt.DecisionTreeRegressor(random_state=0)

predict_linear_regressors(daily_data_by_sample_station, decision_tree_reg)


# ## An extremely randomized tree regressor.
# 
# Extra-trees differ from classic decision trees in the way they are built. When looking for the best split to separate the samples of a node into two groups, random splits are drawn for each of the max_features randomly selected features and the best split among those is chosen. When max_features is set 1, this amounts to building a totally random decision tree.
# 
# Warning: Extra-trees should only be used within ensemble methods.

# In[ ]:


extra_tree_reg = skt.ExtraTreeRegressor(random_state=0)

predict_linear_regressors(daily_data_by_sample_station, extra_tree_reg)


# ## Histogram-based Gradient Boosting Regression Tree.
# This estimator is much faster than GradientBoostingRegressor for big datasets (n_samples >= 10 000).
# 
# This estimator has native support for missing values (NaNs). During training, the tree grower learns at each split point whether samples with missing values should go to the left or right child, based on the potential gain. When predicting, samples with missing values are assigned to the left or right child consequently. If no missing values were encountered for a given feature during training, then samples with missing values are mapped to whichever child has the most samples.
# 
# This implementation is inspired by LightGBM.

# In[ ]:


histogram_based_gradient_boosting_regression_tree_reg = ske.HistGradientBoostingRegressor(min_samples_leaf=1, max_depth=2, learning_rate=1, max_iter=10)

predict_linear_regressors(daily_data_by_sample_station, histogram_based_gradient_boosting_regression_tree_reg)


# ## Gradient Boosting for regression.
# This estimator builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.
# 
# sklearn.ensemble.HistGradientBoostingRegressor is a much faster variant of this algorithm for intermediate datasets (n_samples >= 10_000).

# In[ ]:


gradient_boosting_reg = ske.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='squared_error')

predict_linear_regressors(daily_data_by_sample_station, gradient_boosting_reg)


# ## Prediction voting regressor for unfitted estimators.
# 
# A voting regressor is an ensemble meta-estimator that fits several base regressors, each on the whole dataset. Then it averages the individual predictions to form a final prediction.

# In[ ]:


# reg1 = ske.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='squared_error')
reg2 = sklm.ElasticNet(random_state=0)
reg3 = sklm.LarsCV(cv=5)
reg4 = sklm.RANSACRegressor(random_state=0)
# voting_reg = ske.VotingRegressor(estimators=[("gradient_boosting_regressor", reg1), ("elastic_net", reg2), ("lars_cv", reg3), ("ransac_regressor", reg4)])
voting_reg = ske.VotingRegressor(estimators=[("elastic_net", reg2), ("lars_cv", reg3), ("ransac_regressor", reg4)])

predict_linear_regressors(daily_data_by_sample_station, voting_reg)


# ## RNN
# Recurrent Neural Networks

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

results = seasonal_decompose(daily_data_by_sample_station["GSE_WSE"], model='additive', extrapolate_trend='freq', period=1)
results.plot()


# # Multivariate LSTM Forecast Model

# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import seaborn as sns
from math import sqrt


# ## Preprocess series data to be consumed by model implementing supervised learning
# 1. Create predictor variable columns (X values)
# 2. Create variables to be predicted columns (y values)
# 3. Concatenate columns.
# 4. Drop columns that contain "not a number" values.

# In[ ]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    return agg


# ## Predict recurrent neural network function 
# 1. Predict y value(s) by using x values (test)
# 2. Invert scale for forecast (if scaled was used)
# 3. Invert scale for predictor (if scaled was used)
# 4. Concatenate predictions vs real results in a new data frame
# 5. Plot and print model performance.

# In[ ]:


def predict_recurrent_neural_network(test_ndarrays: List[np.ndarray], model, scaler: skpp=None) -> pd.DataFrame: 
    test_X, test_y = test_ndarrays[2], test_ndarrays[3]
    
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    if scaler is not None:
        inv_yhat = scaler.inverse_transform(inv_yhat)
        print(f'inversed inv_yhat with scale {type(scaler)}')
    inv_yhat = inv_yhat[:,0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    if scaler is not None:
        inv_y = scaler.inverse_transform(inv_y)
        print(f'inversed inv_y with scale {type(scaler)}')
    inv_y = inv_y[:,0]
   
    if scaler is None:
        combined: pd.DataFrame = pd.concat([test.iloc[:-1]["target"], pd.Series(inv_yhat, index=test.iloc[:-1].index)], axis=1)
    else:
        combined: pd.DataFrame = pd.concat([pd.Series(inv_y), pd.Series(inv_yhat)], axis=1)
    combined.columns = ["actual", "predictions"]

    model_metrics(combined["actual"], combined["predictions"])
    combined.plot(figsize=(25,10), fontsize=15)
    
    return combined


# ## Pre process dataset function 
# 1. Create train and test dataframes based on provided dataframe
# 2. Encode labels (optional)
# 3. Cast values to float32 (optional)
# 4. Normalize features with scaler (optional)
# 5. Pre-process data and arrange in the model's supervised learning configuration (using series_to_supervised function)
# 6. Re-shape data to the model's required configuration

# In[ ]:


def pre_process_datasets(dataframe: pd.DataFrame,
                         predictors: List[str]=["WLM_RPE", "WLM_GSE", "RPE_WSE", "GSE_WSE", "WSE"],  
                         scaler: skpp=None, 
                         encode_labels: bool=False, 
                         cast_float32: bool=False) -> []:
    dataframes: List[pd.DataFrame] = dataframe_training_testing_subsets(dataframe)
    train: pd.DataFrame = dataframes[0]
    test: pd.DataFrame = dataframes[1]
    train_values = train[predictors].values
    test_values = test[predictors].values

    # integer encode direction
    if encode_labels:
        encoder = skpp.LabelEncoder()
        train_values[:,3] = encoder.fit_transform(train_values[:,3])
        test_values[:,3] = encoder.fit_transform(test_values[:,3])
        print('encoding labels...')
        print(train_values[:,3])
    
    # ensure all data is float
    if cast_float32:
        train_values = train_values.astype('float32')
        test_values = test_values.astype('float32')
        print('casting to float32...')
        print(train_values[:5])

    # noremalize features
    if scaler is not None:
        train_values = scaler.fit_transform(train_values)
        test_values = scaler.fit_transform(test_values)
        print(f'fitting and transforming to scale {type(scaler)}...')
        print(train_values[:5])

    # frame as supervised learning
    reframed_train = series_to_supervised(train_values, 1, 1)
    reframed_test = series_to_supervised(test_values, 1, 1)

    # drop columsn we don't want to predict
    reframed_train.drop(reframed_train.columns[[5,6,7,9]], axis=1, inplace=True)
    reframed_test.drop(reframed_test.columns[[5,6,7,9]], axis=1, inplace=True)
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X, train_y = reframed_train.values[:, :-1], reframed_train.values[:, -1]
    test_X, test_y = reframed_test.values[:, :-1], reframed_test.values[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    
    # res: List[np.ndarray] = []
    res = []
    res.extend([train_X, train_y, test_X, test_y, scaler])
    return res


# ## Plot history function
# Function to print the model's training history data.

# In[ ]:


from keras.callbacks import History

def plot_history(history: History) -> None:
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


# ## LSTM model 
# Using a sequential pipeline create a LSTM model and add layers to it.
# 1. Train the model
# 2. Plot its history
# 3. Create and print predictions

# In[ ]:


datasets: List[np.ndarray] = pre_process_datasets(daily_data_by_sample_station)
train_X, train_y, test_X, test_y = datasets[0], datasets[1], datasets[2], datasets[3]

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

predict_recurrent_neural_network(datasets, model) 


# In[ ]:


datasets: List[np.ndarray] = pre_process_datasets(daily_data_by_sample_station, scaler=skpp.RobustScaler(), encode_labels=True, cast_float32=True)
train_X, train_y, test_X, test_y, scaler = datasets[0], datasets[1], datasets[2], datasets[3], datasets[4]

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

predict_recurrent_neural_network(datasets, model, scaler) 


# In[ ]:


datasets: List[np.ndarray] = pre_process_datasets(daily_data_by_sample_station, scaler=skpp.MinMaxScaler(), encode_labels=True, cast_float32=True)
train_X, train_y, test_X, test_y, scaler = datasets[0], datasets[1], datasets[2], datasets[3], datasets[4]

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

plot_history(history)
predict_recurrent_neural_network(datasets, model, scaler) 


# In[ ]:


datasets: List[np.ndarray] = pre_process_datasets(daily_data_by_sample_station, scaler=skpp.MaxAbsScaler())
train_X, train_y, test_X, test_y, scaler = datasets[0], datasets[1], datasets[2], datasets[3], datasets[4]

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

plot_history(history)
predict_recurrent_neural_network(datasets, model, scaler) 

