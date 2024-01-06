import requests as req
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta 
from utility import *






def creaet_data_lookup_test(json_file_name='data.json'):
    date = None
    isSubRequest = False
    location_start = None
    is_start = True
    compute_time = 15 #Secound
    priority_FIDs = []
    fuel = 35
    selling_price = 1000
    sugarcane_leaves_per_rai = 1.2
    machine_maintains = {
        "M1": {
            "start_time": "2023"
        }
    }
    
    f = open(f"{json_file_name}")
    datas = json.load(f)
    
    data_set = datas["dataset"]
    user_id =  datas["userid"]
    field_ids =  datas["fields"]
    #print("field_ids", field_ids)
    machine_ids =  datas["machines"]
    date =  datas["date"]
    location_start = json.loads(datas["location_start"])
    
    
    compute_time =  int(datas["calculation_time"])*60 - 15
    fuel =  float(datas["fuel"])
    selling_price =  float(datas["selling_price"])
    #sugarcane_leaves_per_rai = float(datas["sugarcane_leaves_per_rai"])
    

    
    field_ids = json.loads(field_ids)
    machine_ids = json.loads(machine_ids)
    
    #print("machine_ids", machine_ids)
    #for m in machine_ids:
    #    print(m)
    
    
    p1 = None
    p2 = None
    for feild in field_ids:
        feild['id']  = feild['field_id'] 
        if "priority" not in feild:
            continue
        
        if feild['priority'] == 1:
            p1 = feild['id']
        if feild['priority'] == 2:
            p2 = feild['id']
            
    if p2 != None:
        priority_FIDs.append(p2)
    if p1 != None:
        priority_FIDs.insert(0, p1)
    
        

    custom_object = {
    "SELL_PRICE" :selling_price,
    "LEAF_RATE_RAI":sugarcane_leaves_per_rai,
    "OIL_LITER_COST":fuel,
    "IS_CLEAR_ROUTE":True,
    }


    location_start['lat'] = float(location_start['lat'] )
    location_start['lng'] = float(location_start['lng'] )

    #14.602598199645326, 99.73714367756547
    if date == None:
        startdate = datetime(2023, 1, 1)
    else:
        format_string = "%Y-%m-%d"
        startdate = datetime.strptime(date, format_string)

    #print("field_ids", len(field_ids))
    zones = {}
    for feild in field_ids:
        key = feild['zone']
        #print(key)
        if key not in zones:
            zones[key] = {"FID":[], "MID":[]}
        zones[key]["FID"].append(feild['id'])
    for machieid in machine_ids:
        key = machieid['zone']
        if key not in zones:
            continue
        zones[key]["MID"].append(machieid['id'])
        
    zone_id = "Z1"
    fids = zones[zone_id]["FID"]
    mids = zones[zone_id]['MID']
    fzone_datas = []
    mzone_datas = []


    for feild in field_ids:
        if feild['id'] in fids:
            fzone_datas.append(feild)
    for machieid in machine_ids:
        if machieid['id'] in mids:
            mzone_datas.append(machieid)

    #print("location_start", location_start)
    #return {}
    dataset_lookups = get_sub_dataset(user_id, data_set, fzone_datas, custom_object,
                                        mzone_datas, startdate = startdate,
                                        location_start = location_start, is_start = is_start, priority_FIDs=priority_FIDs)
    return dataset_lookups