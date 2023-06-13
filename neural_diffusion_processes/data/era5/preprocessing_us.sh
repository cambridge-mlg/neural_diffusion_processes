#Bash script to preprocess all ERA5 US data.


MIN=1980
MAX=2018
FILEnoYEAR="_ERA5_US"
ROOT="../../../data/era5/us"

pwd

for (( c=$MIN; c<=$MAX; c++ ))
do  
    echo "Year: $c"
    ls -l --block-size=M "${ROOT}/$c"${FILEnoYEAR}.grib

    #Create an unformatted CSV file:
    grib_get_data -p dataDate,dataTime,validityDate,validityTime,shortName "${ROOT}/$c"${FILEnoYEAR}.grib > "${ROOT}/$c"${FILEnoYEAR}_unformatted.csv
    echo "Saved unformatted csv file"

    ls -l --block-size=M "${ROOT}/$c"${FILEnoYEAR}_unformatted.csv

    #Format the csv file to a proper table giving per row one measurement:
    python ../pre_processing/build_value_table_era5.py "${ROOT}/$c"${FILEnoYEAR}_unformatted.csv "${ROOT}/$c"${FILEnoYEAR}_per_measurement.csv
    echo "Formatted CSV file to proper table."

    #Remove unformatted csv file:
    rm "${ROOT}/$c"${FILEnoYEAR}_unformatted.csv
    echo "Removed unformatted csv file"

    ls -l --block-size=M "${ROOT}/$c"${FILEnoYEAR}_per_measurement.csv

    #Reshape the data table from "one row per measurement" to "one row per time point"
    python ../pre_processing/build_data_table_era5.py "${ROOT}/$c"${FILEnoYEAR}_per_measurement.csv "${ROOT}/$c"${FILEnoYEAR}_per_time.csv

    #Remove per measurement file:
    rm "${ROOT}/$c"${FILEnoYEAR}_per_measurement.csv
    echo "Removed per_measurement file"

    ls -l --block-size=M "${ROOT}/$c"${FILEnoYEAR}_per_time.csv

    #Compress and process the data table to a pickle file:
    python ../pre_processing/compress_data_table_era5.py "${ROOT}/$c"${FILEnoYEAR}_per_time.csv "${ROOT}/$c"${FILEnoYEAR}.pickle

    #Remove uncompressed file:
    rm "${ROOT}/$c"${FILEnoYEAR}_per_time.csv
    echo "Removed per_time file"

    ls -l --block-size=M "${ROOT}/$c"${FILEnoYEAR}.pickle

done

echo "Process completed"

