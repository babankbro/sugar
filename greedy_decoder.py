import requests as req
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta 
from utility import *
from test_utility import *
from decoder import *


class GreedyDecoder:
    def __init__(self, dataset_lookups):
        self.dataset_lookups = dataset_lookups
        
    def is_intersect(self, machine_maintenace_infos, fid, start_work_time, end_work_time, start_hour,  working_time_machine):
        status = False
        returntime = -1
        for time_machine_maintenace in machine_maintenace_infos:
            start_day = time_machine_maintenace['start_day']
            end_day = time_machine_maintenace['end_day']
            start_time = time_machine_maintenace['start_time']
            end_time = time_machine_maintenace['end_time']
            
            
            end_hour = start_hour + working_time_machine + 1
            
            if start_time >= 13:
                start_time -= 1
            if start_time < start_hour:
                start_time = start_hour
            if start_time > end_hour:
                start_time = end_hour
                
            if end_time > end_hour:
                end_time = end_hour
            if end_time < start_hour:
                end_time = start_hour
            
            if end_time >= 13:
                end_time -= 1
            
            start_maintain_time = (start_day*working_time_machine +  working_time_machine + (start_time - start_hour))*60
            end_maintain_time = (end_day*working_time_machine +  working_time_machine + (end_time - start_hour))*60
            
            
            
            if end_maintain_time < start_work_time:
                continue
            if start_maintain_time > end_work_time:
                continue
            status = True
            returntime = end_maintain_time
            break
        #print(f"{status}, {fid}, swt-ewt: {start_work_time} - {end_work_time}  smt-ent {start_maintain_time}-{end_maintain_time}" )
        return status, returntime
            
        
    def create_route_info(self, machine, machie_id_type, route, open_times, closed_times, isDebug=False):
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        Start_Times = self.dataset_lookups["Start_Time"]
        End_Times = self.dataset_lookups["End_Time"]
        Setup_Times = self.dataset_lookups["Setup_Time"]
        Areas = self.dataset_lookups["Area"]
        DM = self.dataset_lookups["DM"]
        NAME = self.dataset_lookups["Name"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]
        NF = self.dataset_lookups["Number_Feilds_NF"]
        Machine_Types = self.dataset_lookups["Machine_Type"]
        time_ids = self.dataset_lookups["Time_ID"]
        OPEN_DAY = self.dataset_lookups["OPEN_DAY"]
        CLOSED_DAY = self.dataset_lookups["CLOSED_DAY"]
        
        CLOSED_TIMES = self.dataset_lookups["Closed_Time"]
        MACHINE_ID_TYPE_LOOKUP = self.dataset_lookups["MACHINE_ID_TYPE_LOOKUP"]
        MACHINE_NAME_TYPE_LOOKUP = self.dataset_lookups["MACHINE_NAME_TYPE_LOOKUP"]
        machine_name_type = MACHINE_NAME_TYPE_LOOKUP[machie_id_type]
        machine_setup_time = Setup_Times[machine]
        machine_rate = Operation_Rates[machine]
        machine_working_time = End_Times[machine] - Start_Times[machine] -1 
        
        Machine_IDs = self.dataset_lookups["Machine_ID"]
        MACHINE_MAINTANCE = self.dataset_lookups["MACHINE_MAINTANCE"]
        ROUTES = self.dataset_lookups["ROUTE_ID"]
        machine_maintenace_infos = []
        machine_id = Machine_IDs[machine]
        if machine_id in MACHINE_MAINTANCE:
            machine_maintenace_infos = MACHINE_MAINTANCE[machine_id]
            #print("machine_maintenace_infos",machine_maintenace_infos)

        

        START_FEILD_ID = self.dataset_lookups["START_FEILD_ID"]

        
        if len(route) == 0:
            return True, {"machine": machine,
                    "route": [], 
                    "dt": [],
                    "st": [],
                    "ot": [],
                    "et": [],
                    "ds": [],
                     'days':[],
                     "setup_time":machine_setup_time,
                     "delta_day":[],
                     "machine_working_time":machine_working_time
                    }
        
        
        day = 0
        cid = route[0]
        current_id = cid

        #print(f"OPEN_DAY {OPEN_DAY}")
        #print(f"Operation_Rates {Operation_Rates}")
        
        #|machine_ready_time
        machine_ready_time = 0
        st = max(machine_ready_time, open_times[current_id])
        ot = round(Areas[current_id]*60/machine_rate+ machine_setup_time, 2) 
        et = st + ot
        day = (et/60)//machine_working_time
        ct = et
        
        
        route_ids = ROUTES[machine]
        
        if cid not in route_ids and len(machine_maintenace_infos) > 0:
            ismaintenance, end_maintenance_time = self.is_intersect(machine_maintenace_infos, current_id, 
                                                                    st, et, Start_Times[machine], machine_working_time)
            
            if ismaintenance:
                st = end_maintenance_time
                ot = round(Areas[cid]*60/machine_rate + machine_setup_time, 2) 
                et = round(st+ot, 2)
        
        IS_HAS_START =  self.dataset_lookups["IS_HAS_START"]
        if not IS_HAS_START:
            START_FEILD_ID = current_id
        start_dis = DM[START_FEILD_ID][current_id] 
        dt = DM[START_FEILD_ID][current_id]*60/TRAVELING_SPEED
        dts = [dt]
        sts = [st ]
        ots = [ot]
        ets = [et]
        dss = [start_dis]
        days = [day]
     
        delta_days = [et - day*60*machine_working_time]

        if et > closed_times[route[0]]:
            if isDebug:
                pass
                #print('0000000000000000000000000000000000000000000000000000000000000000000000000000')
            return False, {"machine": machine,
                    "route": [], 
                    "dt": [],
                    "st": [],
                    "ot": [],
                    "et": [],
                    "ds": [],
                     'days':[],
                     "setup_time":machine_setup_time,
                     "delta_day":[],
                     "machine_working_time":machine_working_time
                    }
            
        
        route_info ={"machine": machine,
                    "route": route, 
                    "dt": dts,
                    "st": sts,
                    "ot": ots,
                    "et": ets,
                    "ds": dss,
                     'days':days,
                     "setup_time":machine_setup_time,
                     "delta_day":delta_days,
                     "machine_working_time":machine_working_time
                    }
        #print(route_info)
        
        max_time = np.max(closed_times)
        
        isError = False
        #print("new day cid{0} st {1}, ot {2}, et {3}".format(cid, st, ot, et))
        for i in range(1, len(route)):
            cid = route[i]
            dt = round(DM[current_id][cid]*60/TRAVELING_SPEED, 2)
            ds = DM[current_id][cid]
            machine_ready_time = dt + et
            st = round(max(open_times[cid], machine_ready_time), 2)
            ot = round(Areas[cid]*60/machine_rate + machine_setup_time, 2) 
            et = round(st+ot, 2)
            
            day = (et/60)//machine_working_time
            
            if et > closed_times[cid] :
                isError = True
                if isDebug:
                    pass
                    #print("Error closed_times",  cid, et, closed_times[cid])
                break
            
            if et > max_time:
                isError = True
                #print("Overtime")
                #print("Error max", et, max_time, route[:i])
                break

            
            if cid not in route_ids and len(machine_maintenace_infos) > 0:
                if isDebug:
                    print("===================================", route_ids)
           
                ismaintenance, end_maintenance_time = self.is_intersect(machine_maintenace_infos, cid, st, et, Start_Times[machine], machine_working_time)
                
                if ismaintenance:
                    if isDebug:
                        print("ismaintenance ===================================", route_ids)
                    st = end_maintenance_time
                    ot = round(Areas[cid]*60/machine_rate + machine_setup_time, 2) 
                    et = round(st+ot, 2)

                #if cid == 6 and machine==0:
                    #print(f"et > closed_times[cid] {et} {closed_times[cid]}")
                
                
            dts.append(dt)
            sts.append(st)
            ots.append(ot)
            ets.append(et)
            dss.append(ds)
            days.append(day)
            current_id = cid
            ct = et
            delta_days.append(et - day*60*machine_working_time)
        
        if len(dts) != len(ets):
            if isDebug:
                pass
                print(f"len(dts) != len(ets): {dts} {ets}")
            return False, route_info

        if ets[-1] > max_time : 
            #print(f"et > max_time: {et} {max_time}")
            if isDebug:
                pass
                print(f"et----------------- > max_time: {route} {et} {max_time}")
            return False, route_info
        else:
            return not isError, route_info
        
        
    def create_machine_infos(self, machines, machie_id_type, isDebug=False):
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        Start_Times = self.dataset_lookups["Start_Time"]
        End_Times = self.dataset_lookups["End_Time"]
        machine_infos = {}
        for m in range(len(machines)):
            machine = machines[m]
            MACHINE_NAME_TYPE_LOOKUP = self.dataset_lookups["MACHINE_NAME_TYPE_LOOKUP"]
            machine_name_type = MACHINE_NAME_TYPE_LOOKUP[machie_id_type]
            open_day_times =  np.copy(self.dataset_lookups["OPEN_TYPE_TIMES"][machine_name_type] )
            closed_day_times = np.copy( self.dataset_lookups["CLOSED_TYPE_TIMES"][machine_name_type])
            offset_times = np.copy(self.dataset_lookups["OFSSET_TIMES"][machine_name_type])
            machine_rate = Operation_Rates[machine]
            machine_working_time = End_Times[machine] - Start_Times[machine] -1 #remove break time
            if isDebug:
                pass
                #print("machine_working_time", machine_working_time, open_day_times, closed_day_times)
                
            if machine_working_time < 0:
                print("SHloudddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd not")
                machine_working_time = 8
            
            idx = list(np.where(offset_times >= machine_working_time*60*0.85)[0])
            open_day_times[idx] += 1
            offset_times[idx] = 0

            open_times =  open_day_times*machine_working_time*60 + offset_times
            closed_times = closed_day_times*machine_working_time*60
            
            
            machine_info ={
                "machine_id" : machine,
                "open_time":open_times,
                "closed_time":closed_times
            }
            machine_infos[machine] = (machine_info)
            
        return machine_infos
    
    def get_machine_cost(self, route_info):
        #LEAF_RATE_RAI = self.dataset_lookups["LEAF_RATE_RAI"]
        SELL_PRICE = self.dataset_lookups["SELL_PRICE"]
        OIL_LITER_COST = self.dataset_lookups["OIL_LITER_COST"]
        machine = route_info['machine']
        machine_cost = {"machine_id":'M1', 'machine_type':"Sweeper", 'labor_cost':100, 'maintain_cost':100, 
                        'transport_cost':100, 'operation_cost':100, 'total_cost':400, "total_area":0}
        total_operation_time = np.sum(route_info['ot'])/60
        total_area = np.sum(self.dataset_lookups["Area"][route_info['route']])
        total_leaf_ton = np.sum(self.dataset_lookups["Leaf"][route_info['route']])
        labor_rate_ton = self.dataset_lookups["Labor_Cost"][machine]
        maintain_rate_hour = self.dataset_lookups["Maintain_Cost"][machine]
        total_distance = np.sum(route_info['ds'])
        travel_liter_rate_km = self.dataset_lookups["Fuel_Travel_KM_Rate"][machine]
        operate_liter_rate_ton = self.dataset_lookups["Fuel_Operate_Ton_Rate"][machine]
        setup_time = self.dataset_lookups["Setup_Time"][machine]/60
        number_of_farm = len(route_info['ot'])
        
        machine_cost['machine_id'] = self.dataset_lookups["Machine_ID"][machine]
        machine_cost['machine_type'] = self.dataset_lookups["Machine_Type"][machine]
        machine_cost['labor_cost'] = float(total_leaf_ton*labor_rate_ton)
        machine_cost['total_operation_time'] = float(total_operation_time)
        machine_cost['maintain_rate_hour'] = float(maintain_rate_hour)
        machine_cost['maintain_cost'] = float(total_operation_time *maintain_rate_hour)
        machine_cost['transport_cost'] = float(total_distance*travel_liter_rate_km*OIL_LITER_COST)
        machine_cost['operation_cost'] = float(total_leaf_ton*operate_liter_rate_ton*OIL_LITER_COST)
        machine_cost['labor_rate_ton'] = float(labor_rate_ton)
        
        
        machine_cost['total_distance'] = float(total_distance)
        machine_cost['travel_liter_rate_km'] = float(travel_liter_rate_km)
        machine_cost['operate_liter_rate_ton'] = float(operate_liter_rate_ton)
        machine_cost['total_area'] = float(total_area)
        machine_cost['total_leaf_ton'] = float(total_leaf_ton)
        machine_cost['setup_time'] = float(setup_time)
        machine_cost['travel_liter']  = machine_cost['total_distance']*machine_cost['travel_liter_rate_km']
        machine_cost['operation_liter'] = machine_cost['total_leaf_ton']*machine_cost['operate_liter_rate_ton']
        machine_cost['total_cost'] = machine_cost['labor_cost'] + machine_cost['maintain_cost'] +  machine_cost['transport_cost'] +machine_cost['operation_cost']
        machine_cost['profit_predict'] = machine_cost['total_leaf_ton']*SELL_PRICE
        return  machine_cost
    
    def findbestRouteAndIndex(self, cid, route_infos, machie_id_type, machine_infos, isDebug=False):
        bestCost = 1000000000000000
        bestK = -1
        bestJ = -1
        isCompleted_ALL = False
        for k in range(len(route_infos)):
            route_info = route_infos[k]
            route = route_info['route'][:]
            machine = route_info['machine']
            open_times = machine_infos[machine]["open_time"]
            closed_times = machine_infos[machine]["closed_time"]
            fixed_id = machine_infos[machine]["fixed_id"]
            #if machine != machine_infos[machine]["machine_id"]:
                #print("machine_infos[machine][]", machine, machine_infos[machine]["machine_id"])
            for j in range(fixed_id, len(route)+1):
                route = route_info['route'][:]
                route.insert(j, cid)
                isCompleted, new_route_info= self.create_route_info(machine, machie_id_type, 
                                                                    route, open_times, closed_times, isDebug)
                isCompleted_ALL = isCompleted_ALL or isCompleted
                machine_cost = self.get_machine_cost(new_route_info)
                total_cost = machine_cost['total_cost'] 
                if isCompleted and bestCost > total_cost:
                    bestCost = total_cost
                    bestK = k
                    bestJ = j
                    if cid == 3  and isDebug:
                        pass
                        #print("total_cost", machie_id_type, cid, k, total_cost)
        if isDebug  and bestK == -1:
            pass
            #print("findbestRouteAndIndex", cid, machie_id_type, fixed_id, bestK, bestJ, bestCost, isCompleted_ALL)
        return bestK, bestJ
    
    def remove_item_route(self, fid, route_info):
        if fid not in route_info['route'] or len(route_info['route']) == 0:
            return
        if  len(route_info['dt']) == 0:
            return
        
        index = route_info['route'].index(fid)
        DM = self.dataset_lookups["DM"]
        if index == 0:
            route_info['dt'].pop(0)
            
            if len(route_info['dt']) != 0:
                route_info['dt'][0] = 0
        elif index == len(route_info['route'])-1:
            
            #route_info['dt'].pop(index)
            pass
        else:
            prev_id, next_id = route_info['route'][index-1], route_info['route'][index+1],
            new_dis = DM[prev_id][next_id]
            #route_info['dt'].pop(index)
            #route_info['dt'][index] = new_dis
        #print(index, fid, len(route_info['st']), route_info['dt'])
        #(index, fid, route_info['st'])
        
        try:
            route_info['route'].pop(index)
            route_info['st'].pop(index)
            route_info['dt'].pop(index)
            route_info['ot'].pop(index)
            route_info['et'].pop(index)
            route_info['ds'].pop(index)
            route_info['days'].pop(index)
            route_info['delta_day'].pop(index)
        except:
            #print("Error", fid, len(route_info['route']), index, route_info['st'])
            #print("    ", route_info['route'])
            """
            print(f"pop {fid}: {len(route_info['st'])}" +
                  f"   : {len(route_info['dt'])}" +
                  f"   : {len(route_info['ot'])}" +
                  f"   : {len(route_info['et'])}" +
                  f"   : {len(route_info['ds'])}" +
                  f"   : {len(route_info['days'])}" +
                  f"   : {len(route_info['delta_day'])}")
            """
            pass
        
    def decode_vehicle_type(self, machie_id_type, order_machines, x, 
                            open_set, isDebug=False):
   
        NF = self.dataset_lookups["Number_Feilds_NF"]
        Machine_Types = self.dataset_lookups["Machine_Type"]
        PRIORITY_FEILDS = self.dataset_lookups["PRIORITY_FEILDS"]
        ROUTES = self.dataset_lookups["ROUTE_ID"]
        
        MACHINE_NAME_TYPE_LOOKUP = self.dataset_lookups["MACHINE_NAME_TYPE_LOOKUP"]
        machine_name_type = MACHINE_NAME_TYPE_LOOKUP[machie_id_type]

        machines = list(np.where(Machine_Types == machine_name_type)[0])
        route_infos = []
        close_set = []
        index_route_machine = {}
        machine_infos = self.create_machine_infos(order_machines, machie_id_type, isDebug)
        k = 0
        if isDebug :
            pass
            #print(f" machine_infos {machie_id_type} {machie_id_type}--------------------------------------------------------------------------------")
        for m in order_machines:
            open_times = machine_infos[m]["open_time"]
            closed_times = machine_infos[m]["closed_time"]
            machie_id =  machine_infos[m]["machine_id"]
            index_route_machine[machie_id] = k
            k += 1
            if isDebug and machie_id_type==1:
                pass
                #print("open_times", open_times)
                #print("machine_infos", machie_id,  closed_times)
                #print("diff", machie_id, closed_times)
            route_ids = ROUTES[machie_id]
            isCompleted, new_route_info = self.create_route_info(machie_id, machie_id_type, route_ids[:], 
                                                                 open_times, closed_times)
            route_infos.append(new_route_info)
            machine_infos[machie_id]["fixed_id"] = len(route_ids)
            for rid in route_ids:
                open_set.remove(rid)
            #if isDebug:
                #print(machie_id, Machine_ID[machie_id])
        #print(route_infos)

        cpid = 0
        for i in range(len(PRIORITY_FEILDS)):
            pid = PRIORITY_FEILDS[i]
            if pid in open_set:
                open_set.remove(pid)
                open_set.insert(cpid, pid)
                cpid+=1
            
        #print("Step 2:", open_set)
        
        N = len(open_set)    
        indx =  np.argsort( x[:N])
        open_set = list(np.array(open_set)[indx])
        
        if isDebug:
            pass
            print("PRIORITY_FEILDS", PRIORITY_FEILDS)
            print("open_set", len(open_set), open_set)
        
        
        notfound = 0
        for cid in open_set:
            best_route_index, best_index =  self.findbestRouteAndIndex(cid, route_infos, machie_id_type, machine_infos, isDebug)
            
            if isDebug and cid==0 and machie_id_type == 1:
                #print("isDebug", best_route_index, route_infos[best_route_index]['machine'], route_infos[best_route_index]['route'])
                #print(best_route_index, best_index)
                pass
            
            if best_route_index == -1:
                notfound += 1
                if notfound == 3:
                    break
                continue
            best_route_info = route_infos[best_route_index]
            route = best_route_info['route'][:]
            route.insert(best_index, cid)
            machie_id =  best_route_info['machine']
            open_times = machine_infos[machie_id]["open_time"]
            closed_times = machine_infos[machie_id]["closed_time"]
            isCompleted, new_route_info = self.create_route_info( machie_id, machie_id_type, route, open_times, closed_times, isDebug)
            if isCompleted :
                if isDebug and cid==0 and machie_id_type == 1:
                    pass
                    #print(machie_id, new_route_info, closed_times)
                route_infos[best_route_index] = new_route_info
                #if best_route_index != new_route_info["machine"]:
                    #print("Not mactch", best_route_index, new_route_info["machine"] )
            else:
                print("Errorr----------------------------------------------------ddddddddddddddddddddddddddddddddddddddddddd")
        
        
        #print("Number of Routes", len(route_infos[0]['route']))
        if  isDebug:
            pass
            #print("----------------------------------------------------------------")
            #print(closed_times)
            #for minfo in machine_infos:
                #print(machine_infos[minfo])
        return route_infos
                
        #for i in range():
    def update_openset(self, before_set, next_set, route_infos, extra=[], isDebug=False):
        #print(f"route_infos['route']-->:{len(route_infos[0]['route'])} {route_infos[0]['route']}")
        #print(f"before_set-->:{len(before_set)} {before_set}")
        #print(f"next_set-->:{len(next_set)} {next_set}")
        toRemove=[]
        for item in next_set:
            isFound = False
            for route_info in route_infos:
                if item in route_info['route']:
                    isFound = True
                    break
            if not isFound and not (item in extra) :
                toRemove.append(item)
        for item in toRemove:
            next_set.remove(item)
        #print(f"next_set:{len(next_set)} {next_set}")
        

    def decode(self, x, isDebug=False):
        N = self.dataset_lookups["Number_Feilds"]
        NF = self.dataset_lookups["Number_Feilds_NF"]
        Open_days = self.dataset_lookups["OPEN_DAY"]
    
        self.dataset_lookups["OPEN_TYPE_TIMES"] = {"Sweeper": np.zeros(len(Open_days)), 
                                                   "Baler": np.zeros(len(Open_days)), 
                                                   "Picker": np.zeros(len(Open_days)), 
                                                   "Truck": np.zeros(len(Open_days))}
        self.dataset_lookups["CLOSED_TYPE_TIMES"] = {"Sweeper": np.zeros(len(Open_days)), 
                                                     "Baler": np.zeros(len(Open_days)),
                                                     "Picker": np.zeros(len(Open_days)), 
                                                     "Truck": np.zeros(len(Open_days))} 
        self.dataset_lookups["OFSSET_TIMES"] = {"Sweeper": np.zeros(len(Open_days)), 
                                                "Baler": np.zeros(len(Open_days)),
                                                "Picker": np.zeros(len(Open_days)), 
                                                "Truck": np.zeros(len(Open_days))}
        #print(type(self.dataset_lookups["OPEN_TYPE_TIMES"]))

        meta_infos = {"route_sweep":{}}

        next_open_sweep, next_closed_sweep = get_un_collected(self.dataset_lookups, "Is_Sweep")
        next_open_bale, next_closed_bale = get_un_collected(self.dataset_lookups, "Is_Baled")
        next_open_pick , next_closed_pick= get_un_collected(self.dataset_lookups, "Is_Picked")
        
        if isDebug:
            #print("next_open_sweep", next_open_sweep)
            pass
            #print("next_open_bale", next_open_bale)
            #print("next_open_pick", next_open_pick)
        next_open_sweep_based = next_open_sweep[:]
        next_open_bale_based = next_open_bale[:]
        next_open_pick_based = next_open_pick[:]
        
        M = len(self.dataset_lookups["Machine_ID"])
        mxs = x[-M:]
        
        MACHINE_NAME_TYPE_LOOKUP = self.dataset_lookups["MACHINE_NAME_TYPE_LOOKUP"]
        Machine_Types = self.dataset_lookups["Machine_Type"]
        order_machines = []
        
        NK = 0
        for i in range(4):
            machine_name_type = MACHINE_NAME_TYPE_LOOKUP[i]
            machines = list(np.where(Machine_Types == machine_name_type)[0])
            KI = len(machines)
            #print(i, KI)
            ms = mxs[NK:NK+KI]
            order = np.argsort(ms)
            order_machines.append(machines)
            NK += KI
        
        update_open_closed_time(self.dataset_lookups, "Sweeper", None, isDebug)
        route_infos = self.decode_vehicle_type(0, order_machines[0], x[:N+NF], next_open_sweep[:], isDebug)
        self.update_openset(next_open_sweep, next_open_bale, route_infos, next_closed_sweep)
        meta_infos["route_sweeper"] = route_infos
        
        update_open_closed_time(self.dataset_lookups, "Baler", route_infos, isDebug)
        route_infos = self.decode_vehicle_type(1, order_machines[1], x[(N+NF):(N+NF)*2], next_open_bale[:], isDebug)
        self.update_openset(next_open_bale, next_open_pick, route_infos, next_closed_bale)
        if isDebug:
            print("next_open_pick", next_open_pick, route_infos[0]['route'], next_closed_bale)
        meta_infos["route_baler"] = route_infos
        
        if isDebug:
            pass
            #print("Route Baler  ----------------------- ")
            #for route in route_infos:
                #print(route)
        
        update_open_closed_time(self.dataset_lookups,"Picker", route_infos, isDebug)
        route_infos = self.decode_vehicle_type(2, order_machines[2], x[(N+NF)*2:(N+NF)*3], next_open_pick[:], isDebug)
        meta_infos["route_picker"] = route_infos
        
        machine_name_type = "Truck"
        Machine_Types = self.dataset_lookups["Machine_Type"]
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        machines = list(np.where(Machine_Types == "Truck")[0])
        truck_infos = []
        k = 0
        number_of_routes = []
        for route_type in ["route_sweeper", "route_baler", "route_picker"]:
            for route_info in meta_infos[route_type] :
                #print(route_type, len(route_info['route']), route_info['route'])
                #self.remove_item_route(item, route_info)
                #continue
                number_of_routes.append([route_type, len(route_info['route']), route_info['route']])
        for m in range(len(machines)):
            #print(m, len(order_machines), machines, order_machines, order_machines[3], order_machines[3][m])
            machine =order_machines[3][m]
            if k == len(route_infos):
                break
            truck_info = dict(route_infos[k])
            truck_info['machine'] = machine
            truck_infos.append(truck_info)
            k+=1
        
        meta_infos["route_truck"] = truck_infos
        
        
        cost_datas = {
            "total_income":0,
            "total_labor_cost": 0,
            "total_maintain_cost":0,
            "total_transport_cost":0,
            "total_operation_cost":0,
            "total_cost":0,
            "profit":0,
            "machine_costs":[
                #{"machine_id":'M1', 'machine_type':"Sweeper", 'labor_cost':100, 
                 #                   'maintain_cost':100, 'transport_cost':100, 'total_cost':300},
                #{"machine_id":'M2', 'machine_type':"Baler", 'labor_cost':100, 
                #                    'maintain_cost':100, 'transport_cost':100, 'total_cost':300},
                #{"machine_id":'M3', 'machine_type':"Picker", 'labor_cost':100, 
                 #                   'maintain_cost':100, 'transport_cost':100, 'total_cost':300},
                #{"machine_id":'M4', 'machine_type':"Truck", 'labor_cost':100, 
                       #             'maintain_cost':100, 'transport_cost':100, 'total_cost':300}
            ],
            "farm_incomes":[],
            "farm_areas":[],
            "farm_leafs":[],
        }
        
        
        unused_set = set(next_open_sweep)
        
        for truck_info in truck_infos:
            unused_set.difference_update(truck_info['route'])
        
        
        closed_sets = {"route_sweeper":next_closed_sweep, "route_baler":next_closed_bale , "route_picker":next_closed_pick}
        
        
        
        for item in unused_set:
            #print("unused_set",unused_set)
            for route_type in ["route_sweeper", "route_baler", "route_picker"]:
                if item in closed_sets[route_type]:
                    continue
                for route_info in meta_infos[route_type] :
                    #print(route_info)
                    self.remove_item_route(item, route_info)
                    #pass
        
        total_labor_cost = 0
        total_maintain_cost = 0
        total_transport_cost = 0
        total_operation_cost = 0
        #LEAF_RATE_RAI = self.dataset_lookups["LEAF_RATE_RAI"]
        SELL_PRICE = self.dataset_lookups["SELL_PRICE"]
        OIL_LITER_COST = self.dataset_lookups["OIL_LITER_COST"]

        for route_type in ["route_sweeper", "route_baler", "route_picker", "route_truck"]:
            for route_info in meta_infos[route_type] :
                machine_cost = self.get_machine_cost(route_info)
                total_labor_cost += machine_cost['labor_cost']
                total_maintain_cost += machine_cost['maintain_cost']
                total_transport_cost += machine_cost['transport_cost']
                total_operation_cost += machine_cost['operation_cost']
                cost_datas["machine_costs"].append(machine_cost)
                total_area = machine_cost["total_area"]
                total_leaf_ton = machine_cost["total_leaf_ton"]
                
                if route_type == "route_truck":
                    machine_cost['income'] = total_leaf_ton*SELL_PRICE
                else:
                    machine_cost['income'] = 0
                    
                #if len(route_info["route"]) != len(route_info["dt"]):
                    #print("Error!")

        total_income = 0
        cost_datas["total_labor_cost"] = total_labor_cost
        cost_datas["total_maintain_cost"] = total_maintain_cost
        cost_datas["total_transport_cost"] = total_transport_cost
        cost_datas["total_operation_cost"] = total_operation_cost
        cost_datas["total_cost"] = total_operation_cost + total_labor_cost + total_maintain_cost+total_transport_cost
        
        
        for truck_info in truck_infos:
            for fid in truck_info['route']:
                area = self.dataset_lookups["Area"][fid] #TODO
                leaf = self.dataset_lookups["Leaf"][fid]
                total_income += leaf*SELL_PRICE
                cost_datas["farm_incomes"].append(int(fid))
                cost_datas["farm_areas"].append(area)
                cost_datas["farm_leafs"].append(leaf)
        


        cost_datas["total_income"] = total_income
        cost_datas["profit"] = total_income - cost_datas["total_cost"]

        meta_infos["cost_data"] = cost_datas
        meta_infos['debug'] = {'next_open_sweep':next_open_sweep, 
                               "truck_route":truck_info,
                               "unused_set":unused_set,
                               "number_of_routes":number_of_routes
                               #"name":self.dataset_lookups["Name"][list(unused_set)]
                               }
        
        return meta_infos

if __name__ == "__main__":
    file_name = "data_test/data_U53.json"
    dataset_lookups = creaet_data_lookup_test(file_name)
    print("Machine_ID Length:", len(dataset_lookups["Machine_ID"]))
    update_open_closed_time(dataset_lookups, "Sweeper", None, isDebug=True)
    decoder = GreedyDecoder(dataset_lookups)
    N = dataset_lookups["Number_Feilds"]
    NF = dataset_lookups["Number_Feilds_NF"]
    np.random.seed(0)
    x = np.random.rand(4+ 4*(N+NF))
    meta_infos =decoder.decode(x, True)
    
    #print(rinfos1)
    #meta_infos
    #rinfos1, rinfos2, rinfos3, rinfos4 = meta_infos['route_sweeper'], meta_infos['route_baler'], meta_infos['route_picker'], meta_infos[ 'route_truck']
    
    #display_result_all([rinfos1])
    #display_result_all([rinfos1, rinfos2, rinfos3, rinfos4])
    #print("NF", NF, len(rinfos1[0]['route']))
    #print(meta_infos["cost_data"])
    #routes = dataset_lookups["ROUTE"]
    #decoder.regenerate_route_infos(routes)




