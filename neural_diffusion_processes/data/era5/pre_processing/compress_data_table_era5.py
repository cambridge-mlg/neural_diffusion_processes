import sys
import pandas as pd
import numpy as np

filename_old=sys.argv[1]
filename_new=sys.argv[2]

def compress(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

data=pd.read_csv(filename_old,delimiter=",")

#Remove the dataTime and dataDate column if they are redundant:
#1.Control whether they are redundant and through error if they are not:
valid_time=data.validityTime.to_numpy(dtype=np.int32)
valid_date=data.validityDate.to_numpy(dtype=np.int32)
data_time=data.dataTime.to_numpy(dtype=np.int32)
data_date=data.dataDate.to_numpy(dtype=np.int32)
n_inequal=np.sum(np.logical_or(valid_time!=data_time,valid_date!=data_date))
if n_inequal>0:
    print("Filename: ", filename)
    sys.exit("Error: n_inequal is not zero.")

#2.Remove:
data.drop(columns=["dataDate"],inplace=True) 
data.drop(columns=["dataTime"],inplace=True) 
data.rename(columns={'validityDate': "Date", 'validityTime': "Time"},inplace=True)


#----------TIME COLUMN------------------------
#Format time column to 4-element string:
data['Time']=data['Time'].apply(lambda x: '{0:0>4}'.format(x)).astype(str)
#Format date column to string:
data['Date']=data['Date'].astype(str)
#Insert datetime column:
data.insert(0,"datetime",pd.to_datetime(data['Date']+data['Time'], format='%Y%m%d%H%M'))
#Drop time and data:
data.drop(columns=['Time','Date'],inplace=True)
#---------------------------------------------

#-----------SORT BY TIME---------------------
#Order via datetime:
data=data.sort_values("datetime").reset_index(drop=True)

#Control whether the number of data points per time is equal:
n_total_grid_points_control=data.groupby("datetime").count()['Latitude'][0]

any_inequal=(data.groupby("datetime").count()!=n_total_grid_points_control).any().any()
if any_inequal:
    sys.exit("ERROR: The number of points per time is not equal.")
#---------------------------------------------

#----------FORMAT VARIABLES-----------------------
#Get temperature in Celsius:
data[["2t"]]=data[["2t"]]-273.15
data.rename(columns={'2t': "t_in_Cels" },inplace=True)

#Get pressure in kPa:
data.sp=data.sp/1000
data.rename(columns={'sp': "sp_in_kPa" },inplace=True)

#Rename the wind components:
data.rename(columns={'10u': 'wind_10m_north', 
                    '10v':'wind_10m_east'},inplace=True)
#------------------------------------------------------------


#Compress by choosing right file format:
data=compress(data)

#Save to pickle:                  
data.to_pickle(filename_new)

#Reload data for control:
reloaded_data=pd.read_pickle(filename_new)

#Save:
orig_datetime=data.datetime
rel_datetime=reloaded_data.datetime

#drop time:
data.drop(columns=['datetime'],inplace=True)
reloaded_data.drop(columns=['datetime'],inplace=True)

#Control whether reloading changes anything:
EPS=1e-5
diff_sum_numeric=(data-reloaded_data).abs().sum().sum()
time_change=(orig_datetime!=rel_datetime).any()

if diff_sum_numeric>EPS or time_change:
    sys.exit("Error when reloading file: it seems that the sum of differences is fairly large.")
else:
    print("Compressed data - sample:")
    print(data.sample(10))
