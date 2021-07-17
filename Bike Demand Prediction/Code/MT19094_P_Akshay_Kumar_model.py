import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

train = pd.read_csv("../Datasets/bike_train_final.csv")
test = pd.read_csv("../Datasets/bike_test_final.csv")
train=train.dropna()
test=test.dropna()

#for col in ['Casual', 'Registered']:
#    train['%s_log' % col] = np.log(train[col] + 1)

train['Casual_log'] = np.log(train['Casual'] + 1)
train['Registered_log'] = np.log(train['Registered'] + 1)

date = pd.DatetimeIndex(train['Datetime'])
train['year'] = date.year
train['month'] = date.month
train['hour'] = date.hour
train['dayofweek'] = date.dayofweek
train['hour_workingday_casual'] = train[['hour', 'Working Day']].apply(lambda x: int((x['Working Day'] == 1 and (6 <= x['hour'] <= 17))or (x['Working Day'] == 0 and 6 <= x['hour'] <= 16)), axis=1)
train['hour_workingday_registered'] = train[['hour', 'Working Day']].apply(lambda x: int((x['Working Day'] == 1 and (8 <= x['hour'] <= 17))or (x['Working Day'] == 0 and 5 <= x['hour'] <= 17)), axis=1)

train['isWeekend']=train['dayofweek']// 5 == 1
train['isWeekend']=train['isWeekend'].astype('int')

date = pd.DatetimeIndex(test['Datetime'])
test['year'] = date.year
test['month'] = date.month
test['hour'] = date.hour
test['dayofweek'] = date.dayofweek
test['hour_workingday_casual'] = test[['hour', 'Working Day']].apply(lambda x: int((x['Working Day'] == 1 and (6 <= x['hour'] <= 17))or (x['Working Day'] == 0 and 6 <= x['hour'] <= 16)), axis=1)
test['hour_workingday_registered'] = test[['hour', 'Working Day']].apply(lambda x: int((x['Working Day'] == 1 and (8 <= x['hour'] <= 17))or (x['Working Day'] == 0 and 5 <= x['hour'] <= 17)), axis=1)

test['isWeekend']=test['dayofweek']// 5 == 1
test['isWeekend']=test['isWeekend'].astype('int')

prediction_final = {}

casual_features = ['season_index','isWeekend','Holiday','weather', 'temperature', 'humidity', 'year', 'hour', 'dayofweek', 'hour_workingday_casual']

registered_features = ['season_index','Holiday', 'Working Day', 'weather', 'temperature', 'humidity', 'year', 'hour', 'dayofweek', 'hour_workingday_registered']

#Gradient Boost model on dataset
#gradient_boost=GradientBoostingRegressor(random_state=0, n_estimators=500, min_samples_leaf=5)

error_dict={}

gradient_boost=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                          learning_rate=0.1, loss='ls', max_depth=5,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=50,
                          n_iter_no_change=None, presort='auto',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)

gradient_boost.fit(train[casual_features], train['Casual_log'])
gradient_boost_prediction_casual = gradient_boost.predict(test[casual_features])
gradient_boost_prediction_casual = np.exp(gradient_boost_prediction_casual) - 1
gradient_boost_prediction_casual[gradient_boost_prediction_casual < 0] = 0


gradient_boost.fit(train[registered_features], train['Registered_log'])
gradient_boost_prediction_registered = gradient_boost.predict(test[registered_features])
gradient_boost_prediction_registered = np.exp(gradient_boost_prediction_registered) - 1
gradient_boost_prediction_registered[gradient_boost_prediction_registered < 0] = 0
gradient_boost_prediction_final = gradient_boost_prediction_casual + gradient_boost_prediction_registered
gradient_boost_predicted=pd.DataFrame()
gradient_boost_predicted['Total']=gradient_boost_prediction_final

y_test=test[['Total']]

error_dict['Gradient Boost']=np.sqrt(mean_squared_log_error(y_test,gradient_boost_predicted))

print('RMSLE by Gradient Boost: ',np.sqrt(mean_squared_log_error(y_test,gradient_boost_predicted)))

#Random forest model on dataset
#rfg=RandomForestRegressor(random_state=0, n_jobs=-1,n_estimators=500, min_samples_leaf=5)

rfg=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=9,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=50,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)

rfg.fit(train[casual_features], train['Casual_log'])
rfg_prediction_casual = rfg.predict(test[casual_features])
rfg_prediction_casual = np.exp(rfg_prediction_casual) - 1
rfg_prediction_casual[rfg_prediction_casual < 0] = 0

rfg.fit(train[registered_features], train['Registered_log'])
rfg_prediction_registered = rfg.predict(test[registered_features])
rfg_prediction_registered = np.exp(rfg_prediction_registered) - 1
rfg_prediction_registered[rfg_prediction_registered < 0] = 0

rfg_prediction_final = rfg_prediction_casual + rfg_prediction_registered

rfg_predicted=pd.DataFrame()
rfg_predicted['Total']=rfg_prediction_final

y_test=test[['Total']]

error_dict['Random Forest']=np.sqrt(mean_squared_log_error(y_test,rfg_predicted))
print('RMSLE by Random Forest: ',np.sqrt(mean_squared_log_error(y_test,rfg_predicted)))


xgboost=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=5, min_child_weight=1, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=None, subsample=1, verbosity=1)

#xgboost=xgb.XGBRegressor(learning_rate=0.2,colsample_bytree = 0.4,n_estimators=1000,max_depth=8,gamma=5)

xgboost.fit(train[casual_features], train['Casual_log'])
xgboost_prediction_casual = xgboost.predict(test[casual_features])
xgboost_prediction_casual = np.exp(xgboost_prediction_casual) - 1
xgboost_prediction_casual[xgboost_prediction_casual < 0] = 0

xgboost.fit(train[registered_features], train['Registered_log'])
xgboost_prediction_registered = xgboost.predict(test[registered_features])
xgboost_prediction_registered = np.exp(xgboost_prediction_registered) - 1
xgboost_prediction_registered[xgboost_prediction_registered < 0] = 0


xgboost_prediction_final = xgboost_prediction_casual + xgboost_prediction_registered

xgboost_predicted=pd.DataFrame()
xgboost_predicted['Total']=xgboost_prediction_final

y_test=test[['Total']]

error_dict['XGBoost']=np.sqrt(mean_squared_log_error(y_test,xgboost_predicted))
print('RMSLE by XGBoost: ',np.sqrt(mean_squared_log_error(y_test,xgboost_predicted)))

min_error=min(error_dict.values())

for key,value in error_dict.items():
    if value == min_error:
        best_technique=key
        break

if best_technique=="Gradient Boost":
    gradient_boost_predicted.to_csv("../Output/predicted.csv",index=False)
elif best_technique=="Random Forest":
    rfg_predicted.to_csv("../Output/predicted.csv",index=False)
elif best_technique=="XGBoost":
    xgboost_predicted.to_csv("../Output/predicted.csv",index=False)

plt.title("Regression Technique vs RMSLE")
plt.ylabel("Root mean square logarithmic error")
plt.xlabel("Regression technique")
plt.bar(error_dict.keys(), error_dict.values(),align='center',width = 0.5)
plt.show()