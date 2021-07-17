import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("../Datasets/bike_train_final.csv")
test = pd.read_csv("../Datasets/bike_test_final.csv")
test=test.dropna()

#for col in ['Casual', 'Registered']:
#    train['%s_log' % col] = np.log(train[col] + 1)

train['Total_log'] = np.log(train['Total'] + 1)
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

features = ['season_index','isWeekend','Holiday', 'Working Day', 'weather', 'temperature', 'humidity', 'year', 'hour', 'dayofweek', 'hour_workingday_casual','hour_workingday_registered']

score_dict={}

gradient_boost=GradientBoostingRegressor()
parameters = {'n_estimators': [50,100,150], 'max_depth':[3,5,7,9]}
grid_search = GridSearchCV(estimator=gradient_boost, param_grid=parameters, cv=10, n_jobs=-1)
grid_search.fit(train[features], train['Total_log'])
print(grid_search.best_estimator_)
score_dict['Gradient Boost']=grid_search.best_score_

rfg = RandomForestRegressor()
parameters = {'n_estimators': [50,100,150], 'max_depth':[3,5,7,9]}
grid_search = GridSearchCV(estimator=rfg, param_grid=parameters, n_jobs=1, cv=10)
grid_search.fit(train[features], train['Total_log'])
print(grid_search.best_estimator_)
score_dict['Random Forest']=grid_search.best_score_

xgboost=xgb.XGBRegressor(objective='reg:squarederror')
parameters = {'n_estimators': [50,100,150], 'max_depth':[3,5,7,9]}
grid_search = GridSearchCV(estimator=xgboost, param_grid=parameters, n_jobs=1, cv=10)
grid_search.fit(train[features], train['Total_log'])
print(grid_search.best_estimator_)
score_dict['XGBoost']=grid_search.best_score_

plt.title("Regression Technique vs Best Accuracy Score")
plt.ylabel("Best Accuracy Score")
plt.xlabel("Regression technique")
plt.bar(score_dict.keys(), score_dict.values(),align='center',width = 0.5)
plt.show()
print(score_dict)




