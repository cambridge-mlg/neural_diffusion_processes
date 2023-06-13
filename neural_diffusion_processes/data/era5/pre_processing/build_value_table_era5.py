import pandas as pd
import sys 
'''
This code is taken from the Copernicus Climate data Store website and slighlty modified
to make the CSV file having only one header.
The input is a csv file were the first column has three components: Latitude, Longitude and Time 
(all other variables are saved in different columns).
This code splits this column containing three variables into three different columns.
'''


old_filename=sys.argv[1]
new_filename=sys.argv[2]

#Split the longitude , latitude and the value in different columns:
with open(old_filename, 'r') as f_in, open(new_filename, 'w') as f_out:
    head=next(f_in).replace(" ","")
    f_out.write(head)
    [f_out.write(','.join(line.split()) + '\n') for line in f_in if line[:8]!="Latitude"]