import pandas as pd

def preprocess_holidays():
    #preprocess US holidays list so that it contains only 2017 holidays
    df_holidays=pd.read_csv("../Datasets_orig/usholidays_orig.csv")
    df_holidays['year']=pd.to_datetime(df_holidays['Date'], errors='coerce').dt.year.astype('int')
    df_holidays_final=df_holidays[df_holidays['year']==2017].reset_index()
    df_holidays_final=df_holidays_final[['Date','Holiday']]
    df_holidays_final.to_csv("../Datasets/usholidays.csv",index=False)
    print("US Holidays Dataset preprocessed")

def preprocess_external_factors():
    #preprocess humidity, pressure and wind speed
    external_factors=['humidity','pressure','wind_speed']
    for factor in external_factors:
        df_factor=pd.read_csv("../Datasets_orig/{}_orig.csv".format(factor))
        df_factor['year']=pd.to_datetime(df_factor['Datetime'], errors='coerce').dt.year.astype('int')
        df_factor_final=df_factor[df_factor['year']==2017].reset_index()
        df_factor_final=df_factor_final[['Datetime','Los Angeles']]
        df_factor_final.rename(columns={'Los Angeles': '{}'.format(factor)}, inplace=True)
        df_factor_final.to_csv("../Datasets/{}.csv".format(factor),index=False)
        print("{} Dataset preprocessed".format(factor))

def preprocess_temperature():
    #preprocess temperature and convert temperature in Kelvin to Centigrade
    df_temperature=pd.read_csv("../Datasets_orig/temperature_orig.csv")
    df_temperature['year']=pd.to_datetime(df_temperature['Datetime'], errors='coerce').dt.year.astype('int')
    df_temperature_final=df_temperature[df_temperature['year']==2017].reset_index()
    df_temperature_final=df_temperature_final[['Datetime','Los Angeles']]
    df_temperature_final.rename(columns={'Los Angeles':'temperature'},inplace=True)
    df_temperature_final['temperature']=round(df_temperature_final['temperature']-273,2)
    df_temperature_final.to_csv("../Datasets/temperature.csv",index=False)
    print("Temperature dataset preprocessed")

def preprocess_weather_description():
    df_weather_description=pd.read_csv("../Datasets_orig/weather_description_orig.csv")
    df_weather_description['year']=pd.to_datetime(df_weather_description['Datetime'], errors='coerce').dt.year.astype('int')
    df_weather_description_final=df_weather_description[df_weather_description['year']==2017].reset_index()
    df_weather_description_final=df_weather_description_final[['Datetime','Los Angeles']]
    df_weather_description_final.rename(columns={'Los Angeles':'weather_description'},inplace=True)
    weather_description_dict={'dust':3, 'scattered clouds':2, 'drizzle':4, 'thunderstorm with light rain':4, 'light intensity drizzle':3, 'mist':1, 'fog':2, 'thunderstorm':4, 'proximity thunderstorm':4, 'smoke':3, 'moderate rain':3, 'haze':1, 'light rain':3, 'overcast clouds':2, 'light intensity shower rain':3, 'shower rain':4, 'proximity shower rain':4, 'sky is clear':1, 'broken clouds':2, 'few clouds':3, 'heavy intensity rain':4}
    df_weather_description_final['weather']=df_weather_description_final['weather_description'].replace(weather_description_dict)
    df_weather_description_final.to_csv("../Datasets/weather_description.csv",index=False)
    print("Weather description dataset preprocessed")

def create_season_csv():
    date_range = pd.date_range(start='1/1/2017', end='30/11/2017', freq='H')
    df_season=pd.DataFrame()
    df_season['Datetime']=date_range
    #print(pd.to_datetime(df_season['Datetime'], errors='coerce').dt.month)
    season_dict={1:4, 2:4, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:4}
    df_season['season_index']=pd.to_datetime(df_season['Datetime'], errors='coerce').dt.month.replace(season_dict)
    df_season.to_csv("../Datasets/seasons.csv",index=False)
    print("Season dataset generated")

def preprocess_bike_train_dataset():
    datasets=['q3_2016','q4_2016','q1_2017','q2_2017','q3_2017']

    for dataset in datasets:
        df_train_orig=pd.read_csv("../Datasets_orig/la_bikes_{}_orig.csv".format(dataset))
        df_train_orig=df_train_orig[['start_time','passholder_type']]
        passholder_type_dict={'Monthly Pass':'Registered','Walk-up':'Casual','Flex Pass':'Registered'}
        df_train_final=df_train_orig
        df_train_final['passholder_type']=df_train_orig['passholder_type'].replace(passholder_type_dict)
        df_train_final['Datetime']=pd.to_datetime(df_train_final['start_time'], errors='coerce')
        df_train_final['Datetime']=df_train_final['Datetime'].dt.floor('H')
        df_train_final.drop(['start_time'],axis=1,inplace=True)
        df_train_final=df_train_final[['Datetime','passholder_type']]
        temp_total=df_train_final.groupby('Datetime',as_index=False).count()
        df_train_final=pd.get_dummies(df_train_final, columns=['passholder_type'])
        temp_split=df_train_final.groupby('Datetime',as_index=False).sum()
        df_train_final_updated=pd.merge(temp_total,temp_split,on='Datetime',how='left')
        df_train_final_updated.rename(columns={'passholder_type':'Total'},inplace=True)
        df_train_final_updated.rename(columns={'passholder_type_Casual':'Casual'},inplace=True)
        df_train_final_updated.rename(columns={'passholder_type_Registered':'Registered'},inplace=True)
        df_train_final_updated=df_train_final_updated[['Datetime','Casual','Registered','Total']]
        df_train_final_updated.to_csv("../Datasets/la_bikes_{}_train.csv".format(dataset),index=False)
    print("Training dataset preprocessed")

def preprocess_bike_test_dataset():
    df_test_orig=pd.read_csv("../Datasets_orig/la_bikes_q4_2017_orig.csv")
    df_test_orig=df_test_orig[['start_time','passholder_type']]
    passholder_type_dict={'Monthly Pass':'Registered','Walk-up':'Casual','Flex Pass':'Registered'}
    df_test_final=df_test_orig
    df_test_final['passholder_type']=df_test_orig['passholder_type'].replace(passholder_type_dict)
    df_test_final['Datetime']=pd.to_datetime(df_test_final['start_time'], errors='coerce')
    df_test_final['Datetime']=df_test_final['Datetime'].dt.floor('H')
    df_test_final.drop(['start_time'],axis=1,inplace=True)
    df_test_final=df_test_final[['Datetime','passholder_type']]
    temp_total=df_test_final.groupby('Datetime',as_index=False).count()
    df_test_final=pd.get_dummies(df_test_final, columns=['passholder_type'])
    temp_split=df_test_final.groupby('Datetime',as_index=False).sum()
    df_test_final_updated=pd.merge(temp_total,temp_split,on='Datetime',how='left')
    df_test_final_updated.rename(columns={'passholder_type':'Total'},inplace=True)
    df_test_final_updated.rename(columns={'passholder_type_Casual':'Casual'},inplace=True)
    df_test_final_updated.rename(columns={'passholder_type_Registered':'Registered'},inplace=True)
    df_test_final_updated=df_test_final_updated[['Datetime','Casual','Registered','Total']]
    df_test_final_updated.to_csv("../Datasets/la_bikes_2017_test.csv",index=False)
    print("Testing dataset preprocessed")

if __name__ == '__main__':
    preprocess_holidays()
    preprocess_external_factors()
    preprocess_temperature()
    preprocess_weather_description()
    create_season_csv()
    preprocess_bike_train_dataset()
    preprocess_bike_test_dataset()
    print("Datasets preprocessed")