import pandas as pd

df_train_q1=pd.read_csv("../Datasets/la_bikes_q1_2017_train.csv")
df_train_q2=pd.read_csv("../Datasets/la_bikes_q2_2017_train.csv")
df_train_q3=pd.read_csv("../Datasets/la_bikes_q3_2017_train.csv")

df_train=df_train_q1.append(df_train_q2)
df_train=df_train.append(df_train_q3)

df_holidays=pd.read_csv("../Datasets/usholidays.csv")

df_merged_holidays=pd.merge(df_train,df_holidays,left_on=pd.to_datetime(df_train['Datetime'], errors='coerce').dt.date, right_on=pd.to_datetime(df_holidays['Date']).dt.date, how='left')
df_merged_holidays.drop(['Date'],axis=1,inplace=True)
df_merged_holidays['Holiday']=df_merged_holidays['Holiday'].notnull().astype('int')

df_merged_holidays['Working Day']=((pd.DatetimeIndex(df_merged_holidays['key_0']).dayofweek) // 5 == 0).astype(int)

df_merged_holidays.drop(['key_0'],axis=1,inplace=True)

df_temperature=pd.read_csv("../Datasets/temperature.csv")
df_final=pd.merge(df_merged_holidays,df_temperature,on='Datetime',how='left')

df_humidity=pd.read_csv("../Datasets/humidity.csv")
df_final=pd.merge(df_final,df_humidity,on='Datetime',how='left')

df_wind_speed=pd.read_csv("../Datasets/wind_speed.csv")
df_final=pd.merge(df_final,df_wind_speed,on='Datetime',how='left')

df_weather_description=pd.read_csv("../Datasets/weather_description.csv")
df_final=pd.merge(df_final,df_weather_description,on='Datetime',how='left')

df_seasons=pd.read_csv("../Datasets/seasons.csv")
df_final=pd.merge(df_final,df_seasons,on='Datetime',how='left')

df_pressure=pd.read_csv("../Datasets/pressure.csv")
df_final=pd.merge(df_final,df_pressure,on='Datetime',how='left')

df_final.to_csv("../Datasets/bike_train_final.csv",index=False)

print("Training datasets are merged")

