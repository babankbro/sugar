import requests as req
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import json
import math
URL_DOMAIN = "http://sugar.optzlab.com"

def calculate_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the approximate distance between two points in kilometers using the Haversine formula.
    Arguments:
    lat1 -- latitude of the first point
    lng1 -- longitude of the first point
    lat2 -- latitude of the second point
    lng2 -- longitude of the second point
    """
    earth_radius_km = 6371.0  # Radius of the Earth in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)

    # Calculate the differences between the latitudes and longitudes
    delta_lat = lat2_rad - lat1_rad
    delta_lng = lng2_rad - lng1_rad

    # Apply the Haversine formula
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = earth_radius_km * c

    return distance_km

def base_get_dataset(data_set, user_id, fleids, custom_object, machines,
                     DM, startdate=None, start_id_feild = None, is_start=False, routes=[], lat_lng_start=None, priority_FIDs=[]):
    
    
    for i in range(len(fleids)):
        fleids[i]['is_sweep'] =  1 if fleids[i]['status_sweeper'] == 'finish' else 0
        fleids[i]['is_baled'] = 1 if fleids[i]['status_baler'] == 'finish' else 0
        fleids[i]['is_picked'] = 1 if fleids[i]['status_picker'] == 'finish' else 0
        #fleids[i]['is_sweep'] =  0
        #fleids[i]['is_baled'] = 0
        #fleids[i]['is_picked'] = 0
        #print(fleids[i]['id'], fleids[i]['is_sweep'], fleids[i]['is_baled'], fleids[i]['is_picked'])
        #print(fleids[i]['id'], fleids[i]['status_sweeper'], fleids[i]['status_baler'], fleids[i]['status_picker'])

    #print(query_machine)
    
    
    feild_df = pd.DataFrame(fleids)
    machine_df = pd.DataFrame(machines)
    #print(machine_df)
    #print(machine_df)
    #print("MACHINES", machines)
    #dm_df.to_csv("{0}_{1}.csv".format(data_set, user_id), encoding="utf-8", index=False)
    config_sweep_date = 5

    #print(len(machines), len(feilds))
    feild_df

    LATs = feild_df['LAT'].to_numpy().astype(np.float64)
    LNGs = feild_df['LNG'].to_numpy().astype(np.float64)
    Names = feild_df['name'].to_numpy()
    Areas = feild_df['area_rai'].to_numpy().astype(np.float64)
    Leaf = feild_df['sugarcane_leaves'].to_numpy().astype(np.float64)
    Cut_Dates = feild_df['cut_date'].to_numpy()
    Production_Rates = feild_df['production_rate_ton_per_rai'].to_numpy().astype(np.float64)
    IDs = feild_df['id'].to_numpy()
    Prices = np.ones(len(Production_Rates))*1000

    #LEAF_RATE_RAI = custom_object["LEAF_RATE_RAI"]
    #for i in range(len(Areas)):
        #if Leaf[i] == 0:
            #Leaf[i] = LEAF_RATE_RAI*Areas[i]
    
    #print(feild_df.columns)
    
    if "machine_type_id" in machine_df.columns:
        data = []
        lookup_names = {"1":"sweeper", "2":"baler", "3":"picker", "4":"truck"}
        for i in range(len(machine_df["machine_type_id"])):
            data.append(lookup_names[machine_df.iloc[i]["machine_type_id"]])
        machine_df['machine_type'] = data
        
    #print(machines)

    End_Times = machine_df['end_time'].to_numpy().astype(np.float64)
    Start_Times = machine_df['start_time'].to_numpy().astype(np.float64)
    Setup_Times = machine_df['setup_time'].to_numpy().astype(np.float64)
    Operation_Rates = machine_df['operation_rate_rai_hour'].to_numpy().astype(np.float64)
    #print(Operation_Rates, machine_df.columns)
    Maintain_Costs = machine_df['maintain_cost_baht_hour'].to_numpy().astype(np.float64)
    Machine_Types = machine_df['machine_type'].to_numpy()
    Machine_Names = machine_df['name'].to_numpy()
    Machine_IDs = machine_df['id'].to_numpy()
    Labor_Costs = machine_df['labor_cost'].to_numpy().astype(np.float64)
    Fuel_Travel_Rates = machine_df['fuel_rate_travel_liter_km'].to_numpy().astype(np.float64)
    Fuel_Operate_Rates = machine_df['fuel_operate_liter_ton'].to_numpy().astype(np.float64)
    Schedule_Sweeping_Completed = feild_df['schedule_sweeping_completed'].to_numpy().astype(np.float64)
    Schedule_All_Completions = feild_df['schedule_all_completions'].to_numpy().astype(np.float64)
    SPEED = 20
    
    N = len(Cut_Dates)
    times = []
    
    mintime = datetime.strptime(Cut_Dates[0], '%Y-%m-%d').timestamp()
    maxtime = datetime.strptime(Cut_Dates[0], '%Y-%m-%d').timestamp()
    minI = 0
    maxI = 0
    for i in range(N):
        Cut_Dates[i] = datetime.strptime(Cut_Dates[i], '%Y-%m-%d')
        #print(Cut_Dates[i].timestamp())
        if mintime > Cut_Dates[i].timestamp():
            mintime = Cut_Dates[i].timestamp()
            minI = i
            #print("minI", minI, startdate)
        if mintime < Cut_Dates[i].timestamp():
            maxtime = Cut_Dates[i].timestamp()
            maxI = i
           
        
    for i in range(len(Machine_Types)):
        Machine_Types[i] =  Machine_Types[i][0].upper() +  Machine_Types[i][1:]
    
    if startdate == None:
        startdate = Cut_Dates[minI]
        
        #print(minI)
    else:
        if startdate.timestamp() <= Cut_Dates[minI].timestamp() or startdate.timestamp() > Cut_Dates[maxI].timestamp() :
            startdate = Cut_Dates[minI]

    if start_id_feild == None:
        start_id_feild = IDs[minI]
        print("start_id_feild", start_id_feild, list(np.where(IDs == start_id_feild)[0])[0])


    for i in range(len(DM)):
        DM[i][i] = 0
        #for j in range(len(DM)):
            #DM[i][j] = DM[j][i]

    N = len(Cut_Dates)
    times = []
    Open_days = []
    config_sweep_dates = []
    max_config_days = []
    print("startdate",startdate)
    for i in range(N):
        dt = Cut_Dates[i] - startdate
        hour = dt.total_seconds()/60/60
        Open_days.append(hour/24)
        times.append(hour)
        config_sweep_dates.append(int(Schedule_Sweeping_Completed[i])+1)
        max_config_days.append(int(Schedule_All_Completions[i]))
    AFTER_CUT_DATE = 1
    OPEN_TIMES = np.array(times)
    max_config_days = np.array(max_config_days)
    config_sweep_dates = np.array(config_sweep_dates)
    Open_days = np.array(Open_days) + AFTER_CUT_DATE

    START_GROUP_RATIO = 1
    MAX_WAITING_DAYS = 2
    TRAVELING_SPEED = SPEED
    NF = int(N*START_GROUP_RATIO)
    time_ids = np.argsort(times) 
    CLOSED_TIMES = OPEN_TIMES + config_sweep_dates*24
    Closed_days = Open_days + config_sweep_dates
    #print("Early Set", time_ids)
    #print("Ready Time", OPEN_TIMES[time_ids])
    print(N, NF, len(Cut_Dates))

    idxs = list(np.where(Operation_Rates != 0)[0])
    #print(Operation_Rates, idxs, type(Operation_Rates))
    if (len(Operation_Rates) - len(idxs)) != 0:
        print("---------------------- Found Operation time 0")
        Operation_Rates = Operation_Rates[idxs]
        Maintain_Costs = Maintain_Costs[idxs]
        Machine_Types = Machine_Types[idxs]
        Machine_IDs = Machine_IDs[idxs]
        Machine_Names = Machine_Names[idxs]

    priority_FindexIDs = []
    listIDs = IDs.tolist()
    
    for fid in priority_FIDs:
        if fid in listIDs:
            priority_FindexIDs.append(listIDs.index(fid))
            #pass
    #print(machine_df.columns)
    mantain_infos = {}
    for i in range(len(machine_df)):
        if "machines_maintenance" not in machine_df.columns:
            continue
        mmm = machine_df.iloc[i]
        machine_maintenance= mmm['machines_maintenance']
        print("machine_maintenance", machine_maintenance)
        if len(machine_maintenance) != 0:
            for mm in machine_maintenance:
                machine_id = mm["machine_id"]
                if machine_id not in mantain_infos:
                    mantain_infos[machine_id] = []
                mantain_info = {}    
                date_format = '%d/%m/%Y'
                
                start_date = mm['start_date']
                end_date = mm['end_date']
                start_date = datetime.strptime(start_date, date_format)
                end_date = datetime.strptime(end_date, date_format)
                dt = start_date - startdate
                hour = dt.total_seconds()/60/60
                day = hour/24
                mantain_info['start_day'] = max(day-1, 0)
                dt = end_date - startdate
                hour = dt.total_seconds()/60/60
                day = hour/24
                mantain_info['end_day'] = max(day-1, 0)
                mantain_info['start_time'] = int(mm['start_time'])
                mantain_info['end_time'] = int(mm['end_time'])
                #start_date = start_date + timedelta(hours=22)
                
                mantain_infos[machine_id].append(mantain_info)
                print(mantain_infos)
            
        print(machine_df.iloc[i]['id'], )
        
    key_type_lookups = {"Sweeper": 'is_sweep', "Baler":"is_baled", 
                        "Picker":"is_picked", "Truck":"is_picked"}
    
    routes_ids = {}
    Machine_IDs = list(Machine_IDs)

   
    IDs_ = list(IDs)
    
    for machine_id in routes:
        route_ids = routes[machine_id]
        toremove = []
        for rid in route_ids:
            if rid not in IDs_:
                toremove.append(rid)
        for rid in toremove:
            route_ids.remove(rid)
    
    #routes = []
    for machine_id in routes:
        machine_index = Machine_IDs.index(machine_id)
        mtype = Machine_Types[machine_index]
        key_type = key_type_lookups[mtype]
        #routes[machine_index] =  []
        #continue
        route_ids = [ IDs_.index(fid) for fid in routes[machine_id]]
        
        #print(routes[machine_id], route_ids)
        for i in range(len(fleids)):
            if fleids[i][key_type] == 0 and i in route_ids : 
                route_ids.remove(i)
                
       
        
        #print(machine_index, route_ids)
        routes_ids[machine_index] = route_ids
    
    for i in range(len(machine_df)):
        print(machine_df.iloc[i]['id'], )
        machine_id = machine_df.iloc[i]['id']
        m = Machine_IDs.index(machine_id)
        if m not in routes_ids:
            routes_ids[m] = []
        print(machine_id, m, routes_ids[m])
        
        if custom_object["IS_CLEAR_ROUTE"]:
            routes_ids[m] = []
        
        
    print("custom_object", custom_object)
    #import time
    #time.sleep(10)
    return { 
        "MACHINE_ID_TYPE_LOOKUP" : {"Sweeper": 0, "Baler":1, "Picker":2, "Truck":3},
        "MACHINE_NAME_TYPE_LOOKUP" : {0: "Sweeper", 1:"Baler", 2:"Picker", 3:"Truck"},
        "OPEN_TYPE_TIMES": {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)), "Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))}, 
        "CLOSED_TYPE_TIMES": {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)), "Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))},
        "OFSSET_TIMES": {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)), "Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))}, 
        "CURRENT_DATE":startdate,
        "Lat":LATs,
        "Lng":LNGs,
        "Name":Names,
        "Area":Areas,
        "Leaf":Leaf,
        "Cut_Date":Cut_Dates,
        "OPEN_DAY":Open_days,
        "CLOSED_DAY": Closed_days,
        "Production_Rate":Production_Rates,
        "Feild_ID":IDs,
        "Price":Prices,
        "End_Time":End_Times,
        "Start_Time":Start_Times,
        "Setup_Time":Setup_Times,
        "Operation_Rate":Operation_Rates,
        "Maintain_Cost":Maintain_Costs,
        "Machine_Type":Machine_Types,
        "Machine_ID":Machine_IDs,
        "Machine_Name":Machine_Names,
        "Labor_Cost":Labor_Costs,
        "Fuel_Travel_KM_Rate":Fuel_Travel_Rates,
        "Fuel_Operate_Ton_Rate":Fuel_Operate_Rates,
        "Speed":SPEED,
        "Traveling_Speed":SPEED,
        "DM":DM,
        "Open_Time": OPEN_TIMES,
        "Closed_Time":CLOSED_TIMES,
        "Max_Waiting_Days":MAX_WAITING_DAYS,
        "Number_Feilds":N,
        "Number_Feilds_NF":NF,
        "Time_ID":time_ids,
        "Is_Sweep": feild_df['is_sweep'].to_numpy().astype(np.int32),
        "Is_Baled": feild_df['is_baled'].to_numpy().astype(np.int32),
        "Is_Picked": feild_df['is_picked'].to_numpy().astype(np.int32),
        "SELL_PRICE" :custom_object["SELL_PRICE"],
        "LEAF_RATE_RAI":custom_object["LEAF_RATE_RAI"],
        "OIL_LITER_COST":custom_object["OIL_LITER_COST"],
        "SWEEP_DAY_CONFIG_DAY":config_sweep_dates, #7
        "MAX_DAY_CONFIG_DAY": max_config_days,
        "START_FEILD_ID":start_id_feild,
        "START_LAT_LNG":lat_lng_start,
        "IS_HAS_START": is_start,
        "PRIORITY_FEILDS":priority_FindexIDs,
        "MACHINE_MAINTANCE":mantain_infos,
        "ROUTE":routes,
        "ROUTE_ID":routes_ids,
    }

def get_route_infos(user_id, machines, feilds):
    query_route = f'{URL_DOMAIN}/api/data-get-where.php?db=data_route&where=user_id="{user_id}"'
    print(query_route)
    resp = req.get(query_route)
    routes = resp.json()
    if len(routes) == 0:
        return []
    
    route_df = pd.DataFrame(routes)
    routes = {}
    
    for m in machines:
        machine_id = m['id']
        froute_df = route_df[route_df['machine_id']==machine_id]
        #print(froute_df.head(11))
        route_ids = []
        for k in range(1, len(froute_df['farm_id'])):
            route_ids.append(froute_df.iloc[k]['farm_id'])
        routes[machine_id] =route_ids
    
    for machine_id in routes:
        print(machine_id, routes[machine_id])
    return routes
    


def get_sub_dataset(user_id, data_set, feilds, custom_object, machines, 
                    startdate = datetime(2023, 1, 1), location_start = None, 
                    start_feild_id = None, is_start=False, priority_FIDs=[]):
    
    query_dm = f'{URL_DOMAIN}/DM%20FILE/{data_set}_{user_id}.csv'
    print(query_dm)
    
    dm_df = pd.read_csv(query_dm)
    DM = dm_df.to_numpy()[:, 1:] 
    
    lookup_latlngs = {}
    for fid in feilds:
        fid['id'] = fid['field_id']
        #print(fid)
        lookup_latlngs[fid['id'] ] = [float(fid['LAT']), float(fid['LNG'])]
    feild_df = pd.DataFrame(feilds)
    #feild_df['id'] = feild_df['field_id']

    

    IDs = feild_df['id'].to_numpy()
    DM2 = np.zeros((DM.shape[0]+1, DM.shape[1]+1))
    looku_up_DM_indexs = {}
    for i in range(len(IDs)):
        name = IDs[i]
        k = list(dm_df.columns[1:]).index(name)
        looku_up_DM_indexs[name] = k
    
    for i in range(len(IDs)):
        for j in range(len(IDs)):
            idxi = looku_up_DM_indexs[IDs[i]]
            idxj = looku_up_DM_indexs[IDs[j]]
            DM2[i, j] = DM[idxi, idxj]
    
    start_feild_id = len(IDs) 
    DM2[len(IDs), len(IDs)] = 0
    for i in range(len(IDs)):
        idxi = looku_up_DM_indexs[IDs[i]]
        lat2, lng2 = lookup_latlngs[IDs[i]]
        dis = calculate_distance(location_start['lat'], location_start['lng'], 
                           lat2, lng2) 
        DM2[i, start_feild_id] = dis
        DM2[start_feild_id, i] = dis
        #print(f"Distance {IDs[i]} {dis} {idxi} {start_feild_id}")
    #print(len(IDs),DM2[:, len(IDs)])
    print("start_feild_id", start_feild_id, DM2[9][start_feild_id])
    import time
    #time.sleep(10)

    #print(type(feilds[0]), feild_df['id'])
    #print(type(feilds[0]), feild_df.columns)
    #print("Pirority", priority_FIDs, priority_FindexIDs)
    routes = get_route_infos(user_id, machines, feilds)
    return base_get_dataset(data_set, user_id, feilds, custom_object, machines, DM2, 
                            startdate, start_feild_id, is_start, routes = routes, lat_lng_start=location_start, priority_FIDs=priority_FIDs)
    
    #print("FFF",len(feilds), feilds  )


def get_dataset(user_id, data_set, startdate = datetime(2023, 1, 1),  
                custom_object = {
                    "SELL_PRICE" :1000,
                    "LEAF_RATE_RAI":1.2,
                    "OIL_LITER_COST":35,
                    "IS_CLEAR_ROUTE":True,
                },
                start_feild_id = None, is_start=False, priority_FIDs=[]):
    query_feild = 'https://green.manopna.online/api/data-get-where.php?db=field&where=user_id="{0}"%20and%20data_set="{1}"'.format(user_id, data_set)
    query_machine = 'https://green.manopna.online/api/data-get-where.php?db=machines&where=user_id="{0}"%20and%20data_set="{1}"'.format(user_id, data_set)
    query_dm = 'https://green.manopna.online/DM%20FILE/{0}_{1}.csv'.format(data_set, user_id)

    #feilds = resp.json()
    x = req.get(query_dm)
    dm_df = pd.read_csv(query_dm)
    DM = dm_df.to_numpy()[:, 1:]
    
    
    resp = req.get(query_feild)
    feilds = resp.json()
    print(type(feilds))
    print(type(feilds[0]), feilds[0])
    feild_df = pd.DataFrame(feilds)
    #print("FFF",len(feilds), feilds  )
    resp = req.get(query_machine)
    machines = resp.json()
    IDs = feild_df['id'].to_numpy()
    DM2 = np.zeros(DM.shape)
    looku_up_DM_indexs = {}
    
    
    for i in range(len(DM)):
        name = IDs[i]
        k = list(dm_df.columns[1:]).index(name)
        looku_up_DM_indexs[name] = k

    priority_FindexIDs = []
    for fid in priority_FIDs:
        if fid in looku_up_DM_indexs:
            priority_FindexIDs.append(looku_up_DM_indexs[fid])
    
    for i in range(len(DM)):
        for j in range(len(DM)):
            idxi = looku_up_DM_indexs[IDs[i]]
            idxj = looku_up_DM_indexs[IDs[j]]
            DM2[i, j] = DM[idxi, idxj]
    

    for m in machines:
        if m['id'] == 'MMMM':
            
            m["machines_maintenance"] = [{'id': '35', 'start_date': '4/7/2023', 'start_time': '3', 
                                          'end_date': '7/7/2023', 'end_time': '7', 'time_update': '2023-08-05 08:03:53', 'machine_id': 'M42'}]
            print("MMMMMMMMM",m)
        else:
            m["machines_maintenance"] = []
    print("machine_df", len(machines))
    
    #for m in machines:
        #print(m)
    routes = get_route_infos(user_id, machines, feilds)
    return base_get_dataset(data_set, user_id, feilds, custom_object, machines, DM2,
                            startdate, start_feild_id, is_start, routes=routes, priority_FIDs=priority_FIDs)

temp_json = {
        "solution": "S1",
        "machine_id": "M1",
        "feild_id": "F1",
        "feild_name": "F1",
        "lat": 0,
        "lng": 0,
        "area": 0,
        "machine_type": "Sweeper",
        #"open_time": "08:00",
        "open_date": "2023-01-01",
        "closed_date":"2023-01-01",
        "work_date": "2023-01-01",
        "distance": 0,
        "travel_time": "00:30",
        "start_time": "00:30",
        "operation_time":"00:30",
        "exit_time":"00:30",
    }


def create_update_route_info(k, dataset_lookups, route_info):
    type_names = ["Sweeper", "Baler", "Picker", "Truck"]

    Machine_Types = dataset_lookups["Machine_Type"]
    Setup_Times = dataset_lookups["Setup_Time"]
    CLOSED_TIMES = dataset_lookups["Closed_Time"]
    Cut_Dates = dataset_lookups["Cut_Date"]
    Operation_Rates = dataset_lookups["Operation_Rate"]
    Start_Times = dataset_lookups["Start_Time"]
    End_Times = dataset_lookups["End_Time"]
    IDs = dataset_lookups["Feild_ID"]
    OPEN_TIMES = dataset_lookups["Open_Time"]
    LATs = dataset_lookups["Lat"]
    LNGs = dataset_lookups["Lng"]
    IDs = dataset_lookups["Feild_ID"]
    Areas = dataset_lookups["Area"]
    Leafs = dataset_lookups["Leaf"]
    DM = dataset_lookups["DM"]
    Names = dataset_lookups["Name"]
    route = route_info['route']
    machine_name_type = type_names[k]
    machine = route_info['machine']
    open_day_times =  np.copy(dataset_lookups["OPEN_TYPE_TIMES"][machine_name_type] )
    closed_day_times = np.copy(dataset_lookups["CLOSED_TYPE_TIMES"][machine_name_type])
    offset_times = np.copy(dataset_lookups["OFSSET_TIMES"][machine_name_type])
    machine_rate = Operation_Rates[machine]
    machine_working_time = End_Times[machine] - Start_Times[machine] -1 #remove break time
    START_LAT_LNG = dataset_lookups["START_LAT_LNG"]
    
    idx = list(np.where(offset_times >= machine_working_time*60*0.75)[0])
    open_day_times[idx] += 1
    offset_times[idx] = 0

    open_times =  open_day_times*machine_working_time*60 + offset_times
    closed_times = closed_day_times*machine_working_time*60

    
    update_route_info = []
    prev = route[0]
    last_exit = 0
    arrive = 0
    id = route[0]
    START_FEILD_ID = dataset_lookups["START_FEILD_ID"]
    st = route_info['st'][0] 
    IS_HAS_START =  dataset_lookups["IS_HAS_START"]
    if  IS_HAS_START:
        update_route_info.append({'machine':machine,  'machine type':machine_name_type, "Farm ID":START_FEILD_ID,  
                                            'LAT':START_LAT_LNG['lat'], 'LNG':START_LAT_LNG['lng'], "name": "จุดเริ่มต้น", 'open':0, 
                                            'close':0, 'km':0, "work_day": 0,
                                            "area": 0, 'travel time':0, 'arrive time':arrive, 
                                            'start time':st, "leaf_ton":0,
                                            'operation time': 0, 'end time': st, "working_hour":machine_working_time,
                                            'working_rate':machine_rate, 'day_end_time':0, 
                                            "day_start_time":0})
    #print(route_info)
    #print(len(route_info['dt']), len(route_info['st'])
       # , len(route_info['et']), len(route_info['ot']))
    
    for i in range(len(route)):
        id = route[i]
        name = Names[id]
        
        if i >= len(route_info['dt']):
            print("ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            continue
        
        dt = route_info['dt'][i]
        st = route_info['st'][i] 
        et = route_info['et'][i] 
        ot = route_info['ot'][i]

        arrive = 0
        arrive = (last_exit + dt)
        
        
        if i == 0:
            km = 0
        else:
            km = DM[prev][id]
        area= Areas[id]
        leaf_ton = Leafs[id]
        update_route_info.append({'machine':machine,  'machine type':machine_name_type, "Farm ID":id,  
                                        'LAT':LATs[id], 'LNG':LNGs[id], "name": Names[id], 'open':open_times[id], 
                                         'close':closed_times[id], 'km':km, "work_day": route_info['days'][i],
                                        "area": area, 'travel time':dt, 'arrive time':arrive, 'start time':st, 
                                        'operation time': ot, 'end time': et, "working_hour":machine_working_time,
                                         'working_rate':machine_rate, 'day_end_time':End_Times[machine], 
                                         "day_start_time":Start_Times[machine],
                                         "leaf_ton":leaf_ton})
        last_exit = et
        prev = id
    return update_route_info



temp_json = {
    "solution": "S1",
    "machine_id": "M1",
    "feild_id": "F1",
    "feild_name": "F1",
    "lat": 0,
    "lng": 0,
    "area": 0,
    "machine_type": "Sweeper",
    #"open_time": "08:00",
    "open_date": "2023-01-01",
    "closed_date":"2023-01-01",
    "work_date": "2023-01-01",
    "distance": 0,
    "open_time":"00:30",
    "travel_time": "00:30",
    "start_time": "00:30",
    "operation_time":"00:30",
    "exit_time":"00:30",
}

def convert_update_route_info_to_json(dataset_lookups, update_route_info):
    #temp_json = temp_json.copy()
    IDs = dataset_lookups["Feild_ID"]
    fid = update_route_info["Farm ID"]

    Machine_IDs = dataset_lookups["Machine_ID"]
    
    machine_name_type = update_route_info['machine type']
    current_date = dataset_lookups["CURRENT_DATE"] 
    open_day_times =  np.copy(dataset_lookups["OPEN_TYPE_TIMES"][machine_name_type] )
    closed_day_times = np.copy(dataset_lookups["CLOSED_TYPE_TIMES"][machine_name_type])
    working_hour = update_route_info["working_hour"]
    work_day = update_route_info['work_day']

    start_day = update_route_info['start time']  // (working_hour * 60) 
    start_time = update_route_info["start time"] % (working_hour * 60) 

    exit_day = update_route_info['end time']  // (working_hour * 60) 
    exit_time = update_route_info["end time"] % (working_hour * 60) 
    #print("work_day", work_day, update_route_info['work_day'])
    if fid < len(IDs):
        open_date = (current_date + timedelta(days = open_day_times[fid]) ).strftime("%Y-%m-%d")
        closed_date = (current_date + timedelta(days = closed_day_times[fid]) ).strftime("%Y-%m-%d")
    else:
        open_date = (current_date + timedelta(days = 0 )).strftime("%Y-%m-%d")
        closed_date = (current_date + timedelta(days = 0 )).strftime("%Y-%m-%d")
    work_date = (current_date + timedelta(days = start_day) ).strftime("%Y-%m-%d")
    exit_date = (current_date + timedelta(days = exit_day) ).strftime("%Y-%m-%d")
    #start time
    #print(update_route_info)
    #print(update_route_info['machine'])
    #now.strftime("%m/%d/%Y, %H:%M:%S")
    
    #print(update_route_info['open'], working_hour)
    offset_open = int(update_route_info['open']%((working_hour)*60))
    offset_open = int(update_route_info['open']%((working_hour)*60))
    exit_time_off = update_route_info["day_start_time"] +int(round(exit_time//60, 0)) 
    if exit_time_off >= 12:
        exit_time += 60
    start_time_off= update_route_info["day_start_time"] +int(round(start_time//60, 0)) 
    if start_time_off >= 12:
        start_time += 60



    #print(offset, (working_hour+1), update_route_info['open'], offset//60, offset%60,)
    #offset_start_time = 

    temp_json["machine_id"] = Machine_IDs[update_route_info['machine']]
    if fid < len(IDs):
        temp_json["feild_id"] = IDs[update_route_info['Farm ID']]
    else:
        temp_json["feild_id"] = "SS"
    temp_json["feild_name"] = update_route_info['name']
    temp_json["lat"] = update_route_info['LAT']
    temp_json["lng"] = update_route_info['LNG']
    temp_json["machine_type"] = machine_name_type
    temp_json["open_date"] = open_date
    temp_json["closed_date"] = exit_date 
    temp_json["work_date"] = work_date 
    temp_json["distance"] = update_route_info['km']
    temp_json["open_time"] = "%02d:%02d" % (update_route_info["day_start_time"] + offset_open//60, offset_open%60)
    
    temp_json["travel_time"] = "%02d:%02d" % (int(round(update_route_info["travel time"]//60, 0)) ,
                                              int((round(update_route_info["travel time"] % 60, 0))))
    temp_json["start_time"] = "%02d:%02d" % (update_route_info["day_start_time"] +int(round(start_time//60, 0)) ,
                                            int((round(start_time % 60, 0))))
    temp_json["exit_time"] = "%02d:%02d" % (update_route_info["day_start_time"] +int(round(exit_time//60, 0)) ,
                                            int((round(exit_time % 60, 0))))
    temp_json["operation_time"] = "%02d:%02d" % (int(round(update_route_info["operation time"]//60, 0)) ,
                                            int((round(update_route_info["operation time"] % 60, 0))))

    temp_json["area"] = update_route_info["area"]  
    temp_json["leaf_ton"] = update_route_info["leaf_ton"]  
    temp_json["income"] = update_route_info["leaf_ton"]*dataset_lookups["SELL_PRICE"]
    return dict(temp_json)

def display_route_info(rinfo):
    print(rinfo)
    print("route", rinfo['route'])
    print("tarvel time   :", rinfo['dt'])
    print("start time    :", rinfo['st'])
    print("operation time:", rinfo['ot'])
    print("exit time     :", rinfo['et'])
    print("days     :", rinfo['days'])


def create_json(dataset_lookups, decoder, x):
    update_route_infos = []
    meta_infos =decoder.decode(x)
    rinfos1, rinfos2, rinfos3, rinfos4  = meta_infos['route_sweeper'], meta_infos['route_baler'], meta_infos['route_picker'], meta_infos[ 'route_truck']
    route_infoss = [rinfos1, rinfos2, rinfos3, rinfos4]
    cost_datas =  meta_infos["cost_data"]

    json_datas = []
    df = None
    annots = None
    json_data = dict(temp_json)
    K = 0

    for rinfos in route_infoss:
        #print(K, "==============================", len(rinfos))
        for rinfo in rinfos:
            if len(rinfo['route'] ) == 0:
                continue
            #print( display_route_info(rinfo) )
            update_route_info = create_update_route_info(K, dataset_lookups, rinfo)
            update_route_infos.append(update_route_info)
            #print(update_route_info[0])
           
            for uri in update_route_info:
                #print("----------")
                #print(f"day_start_time: {uri['day_start_time']} travel time: {uri['travel time']}")
                jdata = convert_update_route_info_to_json(dataset_lookups, uri)
                toShow = f'{jdata["machine_id"]}, {jdata["machine_type"]}, {jdata["feild_id"]}, {temp_json["feild_name"]}, ' +\
                         f'{jdata["lat"]}, {jdata["lng"]}, dis: {jdata["distance"]} \n' +\
                         f'[OPEN {jdata["open_date"]} - {jdata["open_time"]} ],[ WORK  {jdata["work_date"]}, {jdata["closed_date"]}, TT: {jdata["travel_time"]}' +\
                         f' ST: {jdata["start_time"]} OT: {jdata["operation_time"]}, ET: {jdata["exit_time"]}'
                #if jdata['machine_id'] == 'M50':
                    #print(f"day_start_time: {uri['day_start_time']} travel time: {uri['start time']} {uri['end time']}")
                    #print(toShow)
                json_datas.append(dict(jdata))
                #j+=1
        #print(len(json_datas))
                
        #break
        K += 1
    #print(meta_infos['debug'])
    
    return update_route_infos, json_datas,cost_datas, df, annots

if __name__ == "__main__":
    user_id = "U16"
    data_set = "D1"
    status = []
    dataset_lookups = get_dataset(user_id, data_set, status, datetime(2023, 4, 4))
    print("Cut_Date", dataset_lookups["Cut_Date"])
    print("OPEN_DAY", dataset_lookups["OPEN_DAY"])
    print("End_Time", dataset_lookups["End_Time"])
    print("Area", dataset_lookups["Area"])
    print("Operation_Rate", dataset_lookups["Operation_Rate"])
    print("Production_Rate", dataset_lookups["Production_Rate"])
    print("Fuel_Travel_KM_Rate", dataset_lookups["Fuel_Travel_KM_Rate"])
    print("Name", dataset_lookups["Name"])
    print("DM", dataset_lookups["DM"][6][7])
    print("IDs", dataset_lookups["Feild_ID"])
    
    #for fd in dataset_lookups["Name"]:
    