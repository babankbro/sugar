import pandas as pd
import matplotlib.pyplot as plt
import utm
import math
import datetime
import numpy as np
import openpyxl


feild_df = pd.read_csv("dataset/feild_v2.csv")
machie_df = pd.read_csv("dataset/machine_v2.csv")
DM_df = pd.read_csv("dataset/DM.csv")
print(feild_df)
print(len(machie_df))

LATs = feild_df['LAT'].to_numpy()
LNGs = feild_df['LNG'].to_numpy()
Names = feild_df['Name'].to_numpy()
Areas = feild_df['Area'].to_numpy()
Cut_Dates = feild_df['Cut Date'].to_numpy()
Production_Rates = feild_df['production rate'].to_numpy()
IDs = feild_df['ID'].to_numpy()



End_Times = machie_df['End Time'].to_numpy()
Start_Times = machie_df['Start Time'].to_numpy()
Setup_Times = machie_df['Setup Time'].to_numpy()
Operation_Rates = machie_df['Operation Rate'].to_numpy()
Maintain_Costs = machie_df['Maintain Cost'].to_numpy()
Machine_Types = machie_df['Machine Type'].to_numpy()
Machine_IDs = machie_df['Machine ID '].to_numpy()
Labor_Costs = machie_df['Labor Cost'].to_numpy()
Fuel_Rates = machie_df['Fuel Rate (travel)'].to_numpy()
N = len(IDs)
Setup_Times = Setup_Times/60

Cut_Dates =  np.array([datetime.datetime(int(x.split('/')[-1]), 
                                         int(x.split('/')[0]), 
                                         int(x.split('/')[1])) for x in Cut_Dates])
print(Cut_Dates)


times = []
startdate = datetime.datetime(2023, 1, 1)


for i in range(N):
    dt = Cut_Dates[i] - startdate
    hour = dt.total_seconds()/60/60
    times.append(hour)
OPEN_TIMES = np.array(times)
print(OPEN_TIMES, "OPEN_TIMES")

DM = DM_df.to_numpy()[:, 1:]
print(DM.shape, np.sum(DM))
#print(DM)
print(machie_df)

