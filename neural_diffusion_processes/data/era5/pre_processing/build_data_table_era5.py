import sys
import pandas as pd

#The short names of the variables which we want to use:
SHORTNAMES=["sp","2t","10u","10v"]


#The arguments give the old and the new filename of the data frame:
filename_old=sys.argv[1]
filename_new=sys.argv[2]

#Read file to data frame:
data=pd.read_csv(filename_old,delimiter=",")

#Permute columns such that shortname is at last:
cols=data.columns.to_list()
cols_perm=cols[:2]+cols[3:]+[cols[2]]
data=data[cols_perm]


#Get a list of data frames per variable--------------------
data_list_single_var=[]
for var in SHORTNAMES:
    #Extract data for certain variable and reset index:
    data_single_var=data[data.shortName==var].reset_index(drop=True)
    #Rename the "value" column to the name of the variable:
    data_single_var.rename(columns={'Value': var },inplace=True)
    #Drop the short name:
    data_single_var.drop(columns=["shortName"],inplace=True) 
    #Append to the list:
    data_list_single_var.append(data_single_var)
#----------------------------------------------------------

#Merge data from different variables-----------------------
merged_data=data_list_single_var[0]
for i in range(1,len(SHORTNAMES)):
    merged_data=pd.merge(merged_data,data_list_single_var[i])
#----------------------------------------------------------

#Control that merged data has the correct number of rows-----
n_rows_merged_data=len(merged_data.index)
n_rows_data=len(data.index)
n_control=int(n_rows_data/7)

if n_control!=n_rows_merged_data:
    print(n_control)
    print(n_rows_merged_data)
    print("Filename old: ", filename_old)
    sys.exit("Error when processing data: Numbers of rows of unprocessed and processed do not fit.")
#------------------------------------------------------------


#Save the file:
merged_data.to_csv(filename_new,index=False)

#Control with reloaded data---------------------------------
reloaded_data=pd.read_csv(filename_new,delimiter=",")
EPS=1e-5
diff_sum=(merged_data-reloaded_data).abs().sum().sum()
if diff_sum>1e-4:
    sys.exit("Error when reloading file: it seems that the sum of differences is fairly large.")
#-----------------------------------------------------------