from asyncio import sleep
#from website import create_app
from flask import render_template, Response
from flask_socketio import SocketIO
import numpy as np
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
#import utm
import math
import requests as req
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from flask import Flask, request
from utility import get_dataset, create_json, get_sub_dataset
from decoder import Decoder
from problem import *
from AMIS import AMIS
from flask_cors import CORS, cross_origin
import flask
import json  
from greedy_decoder import *


#print(LATs)
#print(LNGs)
#plt.figure(figsize=(10, 10))
#plt.scatter(LATs[:], LNGs[:], color='r', s=100)
#plt.savefig('points.png')


from datetime import datetime

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__, template_folder='./')
socketio = SocketIO(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods = ["GET", "POST"])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def home():
    return render_template("index.html")

@app.route("/progress/<socketid>", methods = ["POST"])
async def progress(socketid):


    """
    print("Start progress")
    problem = SurgarProblem(4*(N+NF)+4)
    np.random.seed(0)
    algorithm = AMIS(problem,
        pop_size=100,
        CR=0.3,
        max_iter = 5,
        #dither="vector",
        #jitter=False
    )
    for i in range(algorithm.max_iter):
        socketio.emit("update progress", i *100/algorithm.max_iter , to=socketid)
        #await sleep(1)
        algorithm.single_iterate(i)
    """
    return {
        "greeting": ["hello", "world"],
  
    }

@app.route("/route", methods = ["POST", "GET"])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def cal_route():

    #data_set = request.args.get('dataset')
    #user_id = request.args.get('userid')
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
    
    
    if flask.request.method == 'GET':
        data_set = request.args.get('dataset')
        user_id = request.args.get('userid')
        date =   request.args.get("date")
        url_domain_api = "https://green.manopna.online/api"
        print(data_set, user_id)
        
        query_feild = f'{url_domain_api}/data-get-where.php?db=field&where=user_id="{user_id}"%20and%20data_set="{data_set}"'
        query_machine = f'{url_domain_api}/data-get-where.php?db=machines&where=user_id="{user_id}"%20and%20data_set="{data_set}"'
        query_dm = f'https://green.manopna.online/DM%20FILE/{0}_{1}.csv'.format(data_set, user_id)
        
        
        
    
        resp = req.get(query_feild)
        field_ids = resp.json()

        #print("FFF",len(feilds), feilds  )
        resp = req.get(query_machine)
        machine_ids = resp.json()
        for feild in field_ids:
            #print(feild)
            feild['field_id'] = feild['id'] 
        location_start = {'lat': 14.602598199645326, 'lng': 99.73714367756547}
        
    else: 
        isSubRequest = True
        datas = request.get_json( )
        user_id =  datas["userid"]
        save_file = open(f"/root/sugar_route/data_test/new_data_{user_id}_v3.json", "w")  
        print(save_file)
        json.dump(datas, save_file, indent = 4) 
        save_file.close()  
        
        data_set = datas["dataset"]
        
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
            
            
        #for machine_id in machine_ids:
        #print(machine_id)
        #return
    
    
    #print(field_ids)
    
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

    zones = {}
    for feild in field_ids:
        key = feild['zone']
        if key not in zones:
            zones[key] = {"FID":[], "MID":[]}
        zones[key]["FID"].append(feild['id'])
    for machieid in machine_ids:
        key = machieid['zone']
        if key not in zones:
            continue
        zones[key]["MID"].append(machieid['id'])
    
    #if len(zones) > 1:
        
    """
    if len(zones) > 1:
        print("==============================================================================")
        count_machine_zones = 0
        for key in zones:
            if len(zones[key]["MID"]) > 0:
                count_machine_zones += 1
        if count_machine_zones != len(zones):
            for key in zones:
                zones[key]["MID"] = []
            for machieid in machine_ids:
                for key in zones:
                    if key in machieid['name']:
                        zones[key]["MID"].append(machieid['id'])
                        break
    """

    zone_results = []
    for zone_id in zones:
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
        if len(fzone_datas) == 0 or len(mzone_datas) == 0:
            continue
        
        
        #print("location_start", location_start)
        #return {}
        dataset_lookups = get_sub_dataset(user_id, data_set, fzone_datas, custom_object,
                                            mzone_datas, startdate = startdate,
                                            location_start = location_start, is_start = is_start, priority_FIDs=priority_FIDs)
        
       # route_decoder = Decoder(dataset_lookups)
        route_decoder = GreedyDecoder(dataset_lookups)
        print(dataset_lookups['Feild_ID'])
        print(dataset_lookups['Machine_ID'])

        #for feild in field_ids:
            #print(feild)
        
    
        sugar_problem = SurgarProblem(dataset_lookups)
        sugar_problem.decoder = route_decoder
        np.random.seed(0)

        algorithm = AMIS(sugar_problem,
            pop_size=10, # 1 จำนวนคำตอบต่อรอบ
            CR=0.3,
            max_iter = 1*(len(fzone_datas)+len(mzone_datas)), # 2 จำนวนรอบ
            #max_iter  = 2,
            compute_time = compute_time,
            #max_iter = 10,
            #dither="vector",
            #jitter=False
        )
        print("iterate")
        algorithm.iterate()
        meta_infos = route_decoder.decode(algorithm.bestX, True)
        update_route_infos, json_datas, cost_datas, df, annots = create_json(sugar_problem.dataset_lookups, 
                                                                            route_decoder, algorithm.bestX)

        #meta_infos = route_decoder.decode(algorithm.bestX, True)
        rinfos1, rinfos2, rinfos3, rinfos4  = meta_infos['route_sweeper'], meta_infos['route_baler'], meta_infos['route_picker'], meta_infos[ 'route_truck']
        #for route_info in update_route_infos:
            #print(route_info)
        
        #for j in json_datas:
            #print(j)

        #config_sweep_date = 3
        #print("json_datas",len(json_datas))
        #print("json_datas",len(json_datas[0]), json_datas[0])
        
        #print(field_ids)
        #print(machine_ids)
        #print(date)
        
        for key in   cost_datas:
            if 'numpy' in str(type(cost_datas[key])):
                print(key, type(cost_datas[key]))
                cost_datas[key] = float(cost_datas[key])

        route_infos = [rinfos1, rinfos2, rinfos3, rinfos4]
        #print(dataset_lookups['Feild_ID'])
        for rinfos in route_infos:
            for rinfo in rinfos:
                #pass
                #print(rinfo)
                #print(dataset_lookups['Feild_ID'][rinfo['route']])
                print(rinfo['machine'], dataset_lookups['Feild_ID'][rinfo['route']], rinfo['route'])
        #return json_datas
        #print("Leaf",dataset_lookups["Leaf"])
        if isSubRequest:
            print("IS Post")
        else:
            print("IS Get")
        zone_result = {
            'cost_data':cost_datas,
            'route_data':json_datas
            }
        zone_results.append(zone_result)
    #return {}
    #print(zone_result['cost_data'])
    #x+""
    #print("Leaf",dataset_lookups["Leaf"])
    #print(location_start,  type(location_start))
    #print("startdate", startdate)
    #print("DM",  dataset_lookups['DM'].shape, dataset_lookups['DM'][:, 40])
    #print(dataset_lookups.keys())
    if len(zone_results) == 1:
        print("Zone 1")
        print(zone_results[0]['cost_data'])
        return zone_results[0]
    else:
        zone_result = zone_results[0]
        for i in range(1, len(zone_results)):
            zone_result_2 = zone_results[i]
            cost_datas = zone_result_2['cost_data']
            zone_result['cost_data']["machine_costs"].extend(cost_datas["machine_costs"])
            zone_result['cost_data']["farm_areas"].extend(cost_datas["farm_areas"])
            zone_result['cost_data']["farm_incomes"].extend(cost_datas["farm_incomes"])
            zone_result['cost_data']["profit"] += (cost_datas["profit"])
            zone_result['cost_data']["total_cost"] += (cost_datas["total_cost"])
            zone_result['cost_data']["total_labor_cost"] += (cost_datas["total_labor_cost"])
            zone_result['cost_data']["total_maintain_cost"] += (cost_datas["total_maintain_cost"])
            zone_result['cost_data']["total_operation_cost"] += (cost_datas["total_operation_cost"])
            zone_result['cost_data']["total_transport_cost"] += (cost_datas["total_transport_cost"])

            zone_result['route_data'].extend(zone_result_2['route_data'])
        print("Zone 2")
        return zone_result


if __name__ == "__main__":
    socketio.run(app=app, debug=True, host="0.0.0.0", port = 5014)