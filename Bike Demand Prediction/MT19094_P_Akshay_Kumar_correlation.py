import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("../Datasets/bike_train_final.csv")
test = pd.read_csv("../Datasets/bike_test_final.csv")
test=test.dropna()

date = pd.DatetimeIndex(train['Datetime'])
train['year'] = date.year
train['month'] = date.month
train['hour'] = date.hour
train['dayofweek'] = date.dayofweek + 1
train['hour_workingday_casual'] = train[['hour', 'Working Day']].apply(lambda x: int((x['Working Day'] == 1 and (6 <= x['hour'] <= 17))or (x['Working Day'] == 0 and 6 <= x['hour'] <= 16)), axis=1)
train['hour_workingday_registered'] = train[['hour', 'Working Day']].apply(lambda x: int((x['Working Day'] == 1 and (8 <= x['hour'] <= 17))or (x['Working Day'] == 0 and 5 <= x['hour'] <= 17)), axis=1)

train['isWeekend']=train['dayofweek']// 5 == 1
train['isWeekend']=train['isWeekend'].astype('int')

casual_features = ['Total','Casual','isWeekend','season_index', 'Holiday', 'Working Day', 'weather', 'temperature', 'wind_speed','humidity', 'year', 'hour', 'dayofweek', 'hour_workingday_casual']

casual_corr_matrix = train[casual_features].corr()

f, ax = plt.subplots(figsize =(9, 8))
sns.heatmap(casual_corr_matrix, ax = ax, annot=True)
plt.title("Training dataset correlation - Casual users features")
plt.show()

registered_features = ['Total','Registered','isWeekend','season_index', 'Holiday', 'Working Day', 'weather', 'temperature','wind_speed','humidity', 'year', 'hour', 'dayofweek', 'hour_workingday_registered']

registered_corr_matrix = train[registered_features].corr()

f, ax = plt.subplots(figsize =(9, 8))
sns.heatmap(registered_corr_matrix, ax = ax, annot=True)
plt.title("Training dataset correlation - Registered users features")
plt.show()