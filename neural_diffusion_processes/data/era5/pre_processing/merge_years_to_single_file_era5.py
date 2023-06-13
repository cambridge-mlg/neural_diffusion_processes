import datetime
import sys

import numpy as np
import pandas as pd
import xarray

location=sys.argv[1]
filename_without_year=sys.argv[2]
filename_merged=sys.argv[3]

YEARS_TRAIN=[]
YEARS_VALID=[]
YEARS_TEST=[]
for i in range(1986,2018,4):
    YEARS_TRAIN=YEARS_TRAIN+[i,i+1]
    YEARS_VALID.append(i+2)
    YEARS_TEST.append(i+3)

def merge_df(YEARS_LIST,filename_merged_complete):
    df_list=[]
    for year in YEARS_LIST:
        #Get filename:
        filename=location+str(year)+filename_without_year
        print("Get file: ", filename)
        #Read data from pickle file:
        new_data=pd.read_pickle(filename)
        
        df_list.append(new_data)

    print("Start concatenating:")
    df=pd.concat(df_list)
    print("Finished concatenating.")
    print("Swap columns.")
    #Swap Longitude and Latitutde columns and east and north components of wind such that x1 corresponds to longitutde and x2 to latitude:
    col_list=list(df)
    col_list=[col_list[0]]+[col_list[2]]+[col_list[1]]+col_list[3:5]+[col_list[6]]+[col_list[5]]
    df=df.reindex(columns=col_list).reset_index(drop=True)
    print("Finished swap columns.")
    print("Start sort:")
    df.sort_values(by=['datetime','Longitude','Latitude'], axis=0, inplace=True)
    df.set_index(keys=['datetime','Longitude','Latitude'], drop=True,inplace=True) 
    print("Finished sort.")
    print("Convert to xarray:")
    X=df.to_xarray()
    print("Cast to dtype:")
    X=X.astype(np.float32,casting='same_kind')
    print("Finished cast to dtype.")

    #Save the file:
    print("Convert to netCDF:")
    X.to_netcdf(location+filename_merged_complete+".nc")

merge_df(YEARS_TRAIN,"Test_"+filename_merged)
#merge_df(YEARS_VALID,"Valid_"+filename_merged)
#merge_df(YEARS_TEST,"Test_"+filename_merged)

print("Process finished.")
