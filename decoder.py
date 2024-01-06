import requests as req
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta 
from utility import *
from test_utility import *



def get_un_collected(dataset_lookups, name):
    next_open = []
    next_closed = []
    N = dataset_lookups["Number_Feilds"]
    NF = dataset_lookups["Number_Feilds_NF"]
    for i in range(N):
        if dataset_lookups[name][i] > 2:
            next_closed.append(i)
            continue
        next_open.append(i)
    return next_open, next_closed

def update_open_closed_time(dataset_lookups, machine_type, route_infos, isDebug=True):
    #"OPEN_TYPE_TIMES": {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)), "Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))}, 
    #"CLOSED_TYPE_TIMES": {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)), "Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))}, 
    #if machine_type == "Baler":
    #"OFSSET_TIMES": {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)), "Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))}, 
    Operation_Rates = dataset_lookups["Operation_Rate"]
    Start_Times = dataset_lookups["Start_Time"]
    End_Times = dataset_lookups["End_Time"]
    if machine_type == "Sweeper":
        OPEN_DAY = dataset_lookups["OPEN_DAY"]
        CLOSED_DAY = dataset_lookups["CLOSED_DAY"]
        dataset_lookups["OPEN_TYPE_TIMES"][machine_type] = np.copy(OPEN_DAY)
        dataset_lookups["CLOSED_TYPE_TIMES"][machine_type] = np.copy(CLOSED_DAY)
    if machine_type == "Baler" or machine_type == "Picker" or machine_type == "Truck" :
        OPEN_DAY = dataset_lookups["OPEN_DAY"]
        CLOSED_DAY = dataset_lookups["CLOSED_DAY"]
        dataset_lookups["OPEN_TYPE_TIMES"][machine_type] = np.copy(OPEN_DAY)
        route_ids = []
        for k in range(len(route_infos)):
            
            route_info = route_infos[k]
            #print(route_info)
            machine = route_info['machine']
            
            route_info = route_infos[k]
            route = route_info['route']
            et = np.array(route_info['et'])
            machine_working_time = End_Times[machine] - Start_Times[machine] -1
            if isDebug:
                pass
                #print("########## machine_working_time", machine_working_time, route)
            for jj in range(len(route)):
                j = route[jj]
                route_ids.append(j)
                day = (et[jj]/60)//machine_working_time
                delta = round(et[jj] - day*machine_working_time*60, 2)
                #print(route[j], et[j], day, delta)
                dataset_lookups["OPEN_TYPE_TIMES"][machine_type][j] = day
                dataset_lookups["OFSSET_TIMES"][machine_type][j] = delta
            route_ids.append('|')
        
        MAX_DAY_CONFIG_DAY = dataset_lookups["MAX_DAY_CONFIG_DAY"]
        if machine_type == "Baler":
            MAX_DAY_CONFIG_DAY = MAX_DAY_CONFIG_DAY-1
        
        dataset_lookups["CLOSED_TYPE_TIMES"][machine_type] = CLOSED_DAY + ( MAX_DAY_CONFIG_DAY-
                                                                           dataset_lookups["SWEEP_DAY_CONFIG_DAY"]) 
    if isDebug:
        pass
        """
        if route_infos == None:
            machine = 0
            route_ids = []
            MAX_DAY_CONFIG_DAY = None
        else:
            machine = route_infos[0]['machine']
        machine_working_time = End_Times[machine] - Start_Times[machine] -1
        print("machine_type", machine_working_time, machine_type, route_ids, len(route_ids))
        print(MAX_DAY_CONFIG_DAY, dataset_lookups["SWEEP_DAY_CONFIG_DAY"])
        print("OPEN_TYPE_TIMES", list(dataset_lookups["OPEN_TYPE_TIMES"][machine_type]))
        print("OFSSET_TIMES", list(dataset_lookups["OFSSET_TIMES"][machine_type]))
        print("CLOSED_TYPE_TIMES", list(dataset_lookups["CLOSED_TYPE_TIMES"][machine_type]))
        print("DIF", list(dataset_lookups["CLOSED_TYPE_TIMES"][machine_type] - dataset_lookups["OPEN_TYPE_TIMES"][machine_type]) )
        """


class Decoder:
    def __init__(self, dataset_lookups):
        self.dataset_lookups = dataset_lookups

    def getNext(self, start, ids, closedset, openset, ct, current_id, days, 
                machine_working_time,machine_ready_times, machine_closed_times, 
                machine_rate, machine_setup_time, isDebug):
        ci = start
        M = len(ids)
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]
        Areas = self.dataset_lookups["Area"] #TODO
        max_time = np.max(machine_closed_times)
        min_time = np.min(machine_ready_times)
        isComplete = False
        while ci < M:
            cid = ids[ci]
            ci += 1
            if cid in closedset:
                continue

            next_t = ct + DM[current_id][cid]*60/TRAVELING_SPEED
            if next_t > (days + 1)*machine_working_time:
                next_t = (days + 1)*machine_working_time + DM[current_id][cid]*60/TRAVELING_SPEED
            

           

            if next_t > machine_closed_times[cid]:
                if isDebug:
                    pass
                    #print("2 manage----machine_closed_times--------------------------------", 
                          #next_t, machine_closed_times[cid])
                #openset.append(cid)
                continue
            ot = round(Areas[cid]*60/machine_rate + machine_setup_time, 2) 
            if next_t + ot  > max_time:
                if isDebug:
                    print("3 manage---max_time---------------------------------")
                #openset.append(cid)
                continue
            isComplete = True
            break
        
        if isComplete and ci == len(ids):
            return cid, ci
        
        if ci == len(ids) and (cid in openset):
            if isDebug:
                print("invalid exit", ci,  len(ids), openset)
            return -1, -1
        return cid, ci
    
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
            

    def create_route(self, machine, machie_id_type, fid, close_set, ids,  
                     open_times, closed_times, isDebug):
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        Start_Times = self.dataset_lookups["Start_Time"]
        End_Times = self.dataset_lookups["End_Time"]
        Setup_Times = self.dataset_lookups["Setup_Time"]
        Areas = self.dataset_lookups["Area"]
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]
        NF = self.dataset_lookups["Number_Feilds_NF"]
        Machine_Types = self.dataset_lookups["Machine_Type"]
        Machine_IDs = self.dataset_lookups["Machine_ID"]
        time_ids = self.dataset_lookups["Time_ID"]
        OPEN_DAY = self.dataset_lookups["OPEN_DAY"]
        CLOSED_DAY = self.dataset_lookups["CLOSED_DAY"]
        
        CLOSED_TIMES = self.dataset_lookups["Closed_Time"]
        MACHINE_ID_TYPE_LOOKUP = self.dataset_lookups["MACHINE_ID_TYPE_LOOKUP"]
        MACHINE_NAME_TYPE_LOOKUP = self.dataset_lookups["MACHINE_NAME_TYPE_LOOKUP"]
        machine_name_type = MACHINE_NAME_TYPE_LOOKUP[machie_id_type]
        machine_setup_time = Setup_Times[machine]
        machine_rate = Operation_Rates[machine]
        machine_working_time = End_Times[machine] - Start_Times[machine] -1 #remove break time

        MACHINE_MAINTANCE = self.dataset_lookups["MACHINE_MAINTANCE"]
        machine_id = Machine_IDs[machine]
        machine_maintenace_infos = []
        if machine_id in MACHINE_MAINTANCE:
            machine_maintenace_infos = MACHINE_MAINTANCE[machine_id]
            #print("machine_maintenace_infos",machine_maintenace_infos)

        day = 0
        #print(f"OPEN_DAY {OPEN_DAY}")
        #print(f"Operation_Rates {Operation_Rates}")
        
        
        
        #|machine_ready_time
        machine_ready_time = 0
        st = max(machine_ready_time, open_times[fid])
        ot = round(Areas[fid]*60/machine_rate+ machine_setup_time, 2) 
        et = st + ot
        day = (et/60)//machine_working_time
        #print("V2 new day", day)
        ct = et
        openset = list(ids)
        current_id = fid
        ROUTES = self.dataset_lookups["ROUTE_ID"]
        route_ids = ROUTES[machine]
        
        if current_id not in route_ids and len(machine_maintenace_infos) > 0:
            ismaintenance, end_maintenance_time = self.is_intersect(machine_maintenace_infos, current_id, st, et, Start_Times[machine], machine_working_time)
            
            if ismaintenance:
                st = end_maintenance_time
                ot = round(Areas[current_id]*60/machine_rate + machine_setup_time, 2) 
                et = round(st+ot, 2)
                
        ROUTES = self.dataset_lookups["ROUTE_ID"]
        route = [current_id]
        START_FEILD_ID = self.dataset_lookups["START_FEILD_ID"]
        IS_HAS_START =  self.dataset_lookups["IS_HAS_START"]
        if not IS_HAS_START:
            START_FEILD_ID = current_id
        close_set.append(current_id)
        start_dis = DM[START_FEILD_ID][current_id] 
        dt = DM[START_FEILD_ID][current_id]*60/TRAVELING_SPEED
        dts = [dt]
        sts = [st ]
        ots = [ot]
        ets = [et]
        dss = [start_dis]
        days = [day]
        delta_days = [et - day*60*machine_working_time]
        #if machine == 0:
            #print("ids:", ids, route)
        
        
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

        M = len(ids)
        ci = 1
        dt = 0
        #print(f"IDDDDDDDDDDDDDDDDDDD :{days}")
        while ci < M:
            #print("========================= , ci, ct, st, dt, ot, et", ci, ct, st, dt, ot, et)
            cid, ci = self.getNext(ci, ids, close_set, openset, ct, current_id, day, 
                    machine_working_time, open_times, closed_times, machine_rate, machine_setup_time, isDebug)
            
            
            if cid in close_set:
                ci+=1
                if isDebug:
                    print("cid in close_set", cid, close_set)
                continue

            if cid == -1:
                if isDebug:
                    print("invalid", cid)
                #print("vvvvxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                break
            #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            dt = round(DM[current_id][cid]*60/TRAVELING_SPEED, 2)
            ds = DM[current_id][cid]
            machine_ready_time = dt + et
            st = round(max(open_times[cid], machine_ready_time), 2)
            ot = round(Areas[cid]*60/machine_rate + machine_setup_time, 2) 
            et = round(st+ot, 2)
            max_time = np.max(closed_times)
            
            if cid not in route_ids:
                if  len(machine_maintenace_infos) > 0:
                    ismaintenance, end_maintenance_time = self.is_intersect(machine_maintenace_infos, cid, st, et, Start_Times[machine], machine_working_time)
                    
                    if ismaintenance:
                        st = end_maintenance_time
                        ot = round(Areas[current_id]*60/machine_rate + machine_setup_time, 2) 
                        et = round(st+ot, 2)
                        
                if et > closed_times[cid]:
                    if isDebug:
                        isError = True
                        print("Error", cid, et, closed_times[cid])
                    ci+=1
                    continue

                #print(f"========================= return cid: {cid} ci:{ci} {st} {ot} {et} {closed_times[cid]}  {max_time}")

                if et > max_time:
                    print("Overtime")
                    break
            route.append(cid)
            close_set.append(cid)
            
            dts.append(dt)
            sts.append(st)
            ots.append(ot)
            ets.append(et)
            dss.append(ds)
            current_id = cid
            ct = et
            day = (et/60)//machine_working_time
            days.append(day)
            delta_days.append(et - day*60*machine_working_time)
            #ci += 1

        
            for rid in route_info['route']:
                if rid in openset:
                    openset.remove(rid)
            #print("route", route)
            #print("machine rate", machine_rate)
            #print("working time of machine", machine_working_time)
            #print("travel times", dts)
            #print("start times", sts)
            #print("operation times", ots)
            #print("end times", ets)
            #print("closedset", close_set)
            #print("openset", openset)
            #print()
        
        #print(route_info)

        return route_info, close_set, openset

    def create_route_info(self, machine, machie_id_type, route, open_times, closed_times, isOffset=False):
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
        #print("V2 new day", day)
        ct = et
        
        
        route_ids = ROUTES[machine]
        
        if cid not in route_ids and len(machine_maintenace_infos) > 0:
            ismaintenance, end_maintenance_time = self.is_intersect(machine_maintenace_infos, current_id, st, et, Start_Times[machine], machine_working_time)
            
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
            max_time = np.max(closed_times)
            day = (et/60)//machine_working_time
            
            if cid not in route_ids:
                if  len(machine_maintenace_infos) > 0:
                    ismaintenance, end_maintenance_time = self.is_intersect(machine_maintenace_infos, cid, st, et, Start_Times[machine], machine_working_time)
                    
                    if ismaintenance:
                        st = end_maintenance_time
                        ot = round(Areas[cid]*60/machine_rate + machine_setup_time, 2) 
                        et = round(st+ot, 2)

                #if cid == 6 and machine==0:
                    #print(f"et > closed_times[cid] {et} {closed_times[cid]}")
                if et > closed_times[cid] :
                    isError = True
                    #if machine == 0:
                        #print("Error", NAME[cid], NAME[8], NAME[7], cid, et, closed_times[cid], route[:], ets[:])
                        
                    break

                if et > max_time:
                    #print("Overtime")
                    #print("Error max", et, max_time, route[:i])
                    break
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
            print(f"et > max_time: {et} {max_time}")
            return False, route_info

        if et > max_time : 
            #print(f"et > max_time: {et} {max_time}")
            return False, route_info
        else:
            return not isError, route_info


        #print("route", route)
        #print("machine rate", machine_rate)
        #print("working time of machine", machine_working_time)
        #print("travel times", dts)
        #print("start times", sts)
        #print("operation times", ots)
        #print("end times", ets)

        #print()

    def decode_vehicle_type(self, machie_id_type, order_machines, x, 
                            open_set, isDebug=False, isOffset=False, seed=0):
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        Start_Times = self.dataset_lookups["Start_Time"]
        End_Times = self.dataset_lookups["End_Time"]
        Setup_Times = self.dataset_lookups["Setup_Time"]
        Areas = self.dataset_lookups["Area"]
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]
        NF = self.dataset_lookups["Number_Feilds_NF"]
        Machine_Types = self.dataset_lookups["Machine_Type"]
        time_ids = self.dataset_lookups["Time_ID"]
        PRIORITY_FEILDS = self.dataset_lookups["PRIORITY_FEILDS"]
        ROUTES = self.dataset_lookups["ROUTE_ID"]
        
        CLOSED_TIMES = self.dataset_lookups["Closed_Time"]
        MACHINE_ID_TYPE_LOOKUP = self.dataset_lookups["MACHINE_ID_TYPE_LOOKUP"]
        MACHINE_NAME_TYPE_LOOKUP = self.dataset_lookups["MACHINE_NAME_TYPE_LOOKUP"]
        machine_name_type = MACHINE_NAME_TYPE_LOOKUP[machie_id_type]

        first_ids = np.argsort(x[:NF])
        ids = np.argsort(x[NF:])
        #print("first_ids", first_ids, time_ids[first_ids])
        #print("sort_index", ids)
        
        machines = list(np.where(Machine_Types == machine_name_type)[0])
        #print(seed)
        r1 =np.random.default_rng(int(seed*1000))
        fidx= int(r1.random()*len(first_ids))
        fid = time_ids[first_ids[fidx]]
        np.arange(NF)
        #print(fid, open_set)
        if isDebug:
            print(fid, open_set)
        while fid not in open_set:
            fidx= int(r1.random()*len(first_ids))
            fid = time_ids[first_ids[fidx]]
        
        #open_set = list(ids[:])
        route_infos = []
        close_set = []
        #print("open_times", open_times)
        #print("sort_index", ids)
        if isDebug:
            print("order_machines", order_machines)
        for m in range(len(machines)):
            machine = machines[order_machines[m]]
            
            open_day_times =  np.copy(self.dataset_lookups["OPEN_TYPE_TIMES"][machine_name_type] )
            closed_day_times = np.copy( self.dataset_lookups["CLOSED_TYPE_TIMES"][machine_name_type])
            offset_times = np.copy(self.dataset_lookups["OFSSET_TIMES"][machine_name_type])
            machine_rate = Operation_Rates[machine]
            machine_working_time = End_Times[machine] - Start_Times[machine] -1 #remove break time
            
            idx = list(np.where(offset_times >= machine_working_time*60*0.75)[0])
            open_day_times[idx] += 1
            offset_times[idx] = 0

            open_times =  open_day_times*machine_working_time*60 + offset_times
            closed_times = closed_day_times*machine_working_time*60
            if isDebug:
                print(f"Machine{machine} {machine_name_type} {machine_working_time}")
                print("open_times", open_times)
                print("closed_times", closed_times)
            isFound = False
            for fidx in range(m, 4):
                fid = time_ids[first_ids[fidx]]
                if fid in open_set:
                    isFound = True
                    break
            if not isFound:
                fid = open_set[np.argsort(open_times[open_set])[0]]
                #print("Not found best start", fid)
            
            item = fid
            open_set_subs = list(np.array(open_set)[ list(np.argsort( first_ids[:len(open_set)]))])
            fixed_index = len(PRIORITY_FEILDS)
            last_fixed_index = 0
            
            if machine in ROUTES:
                route_ids = ROUTES[machine]
                if isDebug:
                    print("FIXED ID", machine, route_ids)
                    #print("Befpre" , open_set_subs)
                k = 0
                for rid in route_ids:
                    if rid in open_set_subs:
                        open_set_subs.remove(rid)
                    open_set_subs.insert(k, rid)
                    k += 1
                    
                last_fixed_index = len(route_ids)
                #if isDebug:
                    #print("After" , open_set_subs)

            if len(PRIORITY_FEILDS) > 0:
                fid = PRIORITY_FEILDS[0]
                
                open_set_subs.remove(fid)
                open_set_subs.insert(last_fixed_index, fid)
                if len(PRIORITY_FEILDS) == 2:
                    fid2 = PRIORITY_FEILDS[1]
                    open_set_subs.remove(fid2)
                    open_set_subs.insert(last_fixed_index+1, fid2)
                

            if machine in ROUTES and len(ROUTES[machine]) > 0:
                route_ids = ROUTES[machine]
                fixed_index = open_set_subs.index(route_ids[-1])
                fid = open_set_subs[0]
            
            if fid != open_set_subs[0]:
                fid = open_set_subs[0]

            if isDebug:
                print("PIORITY ", fid, open_set_subs, list(first_ids))
            #print(" openset", open_set_subs, open_set, close_set)
            route_info, close_set, open_set_out  = self.create_route(machine, machie_id_type, fid, close_set,
                                                                     open_set_subs, open_times, closed_times, isDebug)
            if isDebug:
                pass
                #print(f"open_set_out {open_set_out}")
            
            if len(open_set_out) == 0:
                open_set.remove(item)
                #print("remove openset", item, open_set_subs, open_set, close_set)
            else:
                open_set = list(open_set_out)
            #print(" openset 3", open_set)
            if route_info == None:
                print("SHOLD NOT-------------------------------------------------")
            route_infos.append(route_info)

            if len(open_set_out) == 0:
                break
        
        for item in close_set:
            if item in open_set:
                open_set.remove(item)
        #print("last open_setxxxxxxxx", open_set, len(route_infos), close_set, route_infos[0]['route'])
        if isDebug:
            print("last open_setxxxxxxxx", machine,  open_set, len(route_infos), close_set, route_infos[0]['route'])
        count_complete = len(close_set)
        for cid in open_set:
            bestK = -1
            bestJ = -1
            bestCost = 1000000000000000
            bestRouteInfo = None
            for k in range(len(route_infos)):
                route_info = route_infos[k]
                route = route_info['route'][:]
                machine = route_info['machine']
                lastCost = np.sum(route_info['dt'])
                for j in range(fixed_index, len(route)+1):
                    route = route_info['route'][:]
                    route.insert(j, cid)
                    #print("check route", route)
                    isCompleted, new_route_info= self.create_route_info(machine, machie_id_type, route, open_times, closed_times, isOffset)
                    #cost = np.sum(new_route_info['dt']) - 1000000*len(new_route_info['route'])
                    machine_cost = self.get_machine_cost(new_route_info)
                    total_cost = machine_cost['total_cost']
                    cost = total_cost-1000*(machine_cost["total_area"])

                    #print(f"isCompleted {isCompleted}: cost:{cost} {route}")
                    if isCompleted and bestCost > cost:
                        #print(f" {cid} {total_cost} isCompleted {isCompleted}: cost:{cost} {route}")
                        bestCost = cost
                        bestK = k
                        bestJ = j
                        bestRouteInfo = new_route_info
            if bestK == -1:
                #print(cid, "Nnone")
                continue
            if bestRouteInfo == None:
                continue
                
            route_infos[bestK] = bestRouteInfo
            count_complete += 1
            
            #print(cid, bestRouteInfo['route'])
        
        #print("---------------------", count_complete, len(open_set), open_set)
        if count_complete != len(open_set) + len(close_set):
            return False, route_infos
        return True, route_infos
    
    

    def update_openset(self, before_set, next_set, route_infos, extra=[]):
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
        machine_cost['maintain_cost'] = float((total_operation_time - number_of_farm*setup_time)*maintain_rate_hour)
        machine_cost['transport_cost'] = float(total_distance*travel_liter_rate_km*OIL_LITER_COST)
        machine_cost['operation_cost'] = float(total_leaf_ton*operate_liter_rate_ton*OIL_LITER_COST)
        machine_cost['labor_rate_ton'] = float(labor_rate_ton)
        machine_cost['total_operation_time'] = float(total_operation_time)
        machine_cost['maintain_rate_hour'] = float(maintain_rate_hour)
        machine_cost['total_distance'] = float(total_distance)
        machine_cost['travel_liter_rate_km'] = float(travel_liter_rate_km)
        machine_cost['operate_liter_rate_ton'] = float(operate_liter_rate_ton)
        machine_cost['total_area'] = float(total_area)
        machine_cost['total_leaf_ton'] = float(total_leaf_ton)
        machine_cost['setup_time'] = float(setup_time)
        machine_cost['travel_liter']  = machine_cost['total_distance']*machine_cost['travel_liter_rate_km']
        machine_cost['operation_liter'] = machine_cost['total_leaf_ton']*machine_cost['operate_liter_rate_ton']
        

        
        machine_cost['total_cost'] = machine_cost['labor_cost'] + machine_cost['maintain_cost'] +  machine_cost['transport_cost'] +machine_cost['operation_cost']
        return  machine_cost

    def decode(self, x, isDebug=False):
        N = self.dataset_lookups["Number_Feilds"]
        NF = self.dataset_lookups["Number_Feilds_NF"]
        Open_days = self.dataset_lookups["OPEN_DAY"]
    
        self.dataset_lookups["OPEN_TYPE_TIMES"] = {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)), "Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))}
        self.dataset_lookups["CLOSED_TYPE_TIMES"] = {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)),"Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))} 
        self.dataset_lookups["OFSSET_TIMES"] = {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)),"Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))}
        #print(type(self.dataset_lookups["OPEN_TYPE_TIMES"]))

        meta_infos = {"route_sweep":{}}

        next_open_sweep, next_closed_sweep = get_un_collected(self.dataset_lookups, "Is_Sweep")
        next_open_bale, next_closed_bale = get_un_collected(self.dataset_lookups, "Is_Baled")
        next_open_pick , next_closed_pick= get_un_collected(self.dataset_lookups, "Is_Picked")
        
        if isDebug:
            print("next_open_sweep", next_open_sweep)
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
            ms = mxs[NK:NK+KI]
            order = np.argsort(ms)
            order_machines.append(order)
            NK += KI

        #print("next_open Sweep", next_open)
        update_open_closed_time(self.dataset_lookups, "Sweeper", None, isDebug)
        is_complete, route_infos = self.decode_vehicle_type(0, order_machines[0], 
                                                            x[:N+NF], next_open_sweep[:], isDebug, seed=x[4*(N+NF)])
        #print(f"is_complete:{is_complete}")
        if isDebug:
            print("next_open_bale 11", next_open_bale, next_closed_sweep)
            #print("next_open_bale 11", next_open_bale)
        self.update_openset(next_open_sweep, next_open_bale, route_infos, next_closed_sweep)
        meta_infos["route_sweeper"] = route_infos
        #print("next_open Sweep", next_open)
        
        update_open_closed_time(self.dataset_lookups, "Baler", route_infos, isDebug)
        if isDebug:
            print("next_open_bale 22", next_open_bale)
        is_complete, route_infos = self.decode_vehicle_type(1, order_machines[1], x[(N+NF):(N+NF)*2], next_open_bale[:], isDebug, seed=x[4*(N+NF)+1])
        #print(f"is_complete:{is_complete}")
        self.update_openset(next_open_bale, next_open_pick, route_infos, next_closed_bale)
        meta_infos["route_baler"] = route_infos
        update_open_closed_time(self.dataset_lookups,"Picker", route_infos, isDebug)
        
        if isDebug:
            print(route_infos[0]['route'])
            #self.update_openset(next_open_bale, next_open_pick, route_infos)
            #print("next_open_sweep", next_open_sweep)
            #print("next_open_bale", next_open_bale)
            #print("next_open_pick", next_open_pick)
        
        is_complete, route_infos = self.decode_vehicle_type(2, order_machines[2], x[(N+NF)*2:(N+NF)*3], next_open_pick[:], isDebug, seed=x[4*(N+NF)+2])
        #self.update_openset(next_open_bale, next_open_pick, route_infos)
        #print(f"is_complete:{is_complete}")
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
            machine = machines[order_machines[3][m]]
            
            open_day_times =  np.copy(self.dataset_lookups["OPEN_TYPE_TIMES"][machine_name_type] )
            closed_day_times = np.copy( self.dataset_lookups["CLOSED_TYPE_TIMES"][machine_name_type])
            offset_times = np.copy(self.dataset_lookups["OFSSET_TIMES"][machine_name_type])
            #machine_rate = Operation_Rates[machine]
            #machine_working_time = End_Times[machine] - Start_Times[machine] -1 #remove break time

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
    

    def regenerate_route_infos(self, routes, isDebug=True):
        IDs = list(self.dataset_lookups["Feild_ID"])
        Machine_IDs = list(self.dataset_lookups["Machine_ID"])
        Machine_Types = list(self.dataset_lookups["Machine_Type"])
        N = self.dataset_lookups["Number_Feilds"]
        NF = self.dataset_lookups["Number_Feilds_NF"]
        Open_days = self.dataset_lookups["OPEN_DAY"]
    
        self.dataset_lookups["OPEN_TYPE_TIMES"] = {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)), "Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))}
        self.dataset_lookups["CLOSED_TYPE_TIMES"] = {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)),"Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))} 
        self.dataset_lookups["OFSSET_TIMES"] = {"Sweeper": np.zeros(len(Open_days)), "Baler": np.zeros(len(Open_days)),"Picker": np.zeros(len(Open_days)), "Truck": np.zeros(len(Open_days))}
        #print(type(self.dataset_lookups["OPEN_TYPE_TIMES"]))

        meta_infos = {"route_sweep":{}}

        next_open_sweep, next_closed_sweep = get_un_collected(self.dataset_lookups, "Is_Sweep")
        next_open_bale, next_closed_bale = get_un_collected(self.dataset_lookups, "Is_Baled")
        next_open_pick , next_closed_pick= get_un_collected(self.dataset_lookups, "Is_Picked")
        
        if isDebug:
            print("next_open_sweep", next_open_sweep)
            print("next_open_bale", next_open_bale)
            print("next_open_pick", next_open_pick)
        next_open_sweep_based = next_open_sweep[:]
        next_open_bale_based = next_open_bale[:]
        next_open_pick_based = next_open_pick[:]
        print(Machine_Types)
        order_machines = []
        MACHINE_NAME_TYPE_LOOKUP = self.dataset_lookups["MACHINE_NAME_TYPE_LOOKUP"]
        
   
        
        order_machine_types = ['Sweeper', 'Baler', 'Picker', 'Truck']
        for mtype in order_machine_types:
            
            order_machines = []
            route_machines = []
            for machine_id in routes:
            
                machine_index = Machine_IDs.index(machine_id)
                if Machine_Types[machine_index] != mtype:
                    continue
                route_ids = [ IDs.index(fid) for fid in routes[machine_id]]
                #print(machine_index, route_ids)
                order_machines.append(machine_index)
                route_machines.append(route_ids)
            print(mtype, order_machines)
            if mtype == "Sweeper":
                update_open_closed_time(self.dataset_lookups,"Sweeper", None, isDebug)
                is_complete, route_infos = self.decode_vehicle_type2(0, order_machines, route_machines, next_open_sweep[:])
                self.update_openset(next_open_sweep, next_open_bale, route_infos, next_open_sweep_based)
                meta_infos["route_sweeper"] = route_infos
            elif mtype == "Baler":
                update_open_closed_time(self.dataset_lookups, "Baler", route_infos, isDebug)
                is_complete, route_infos = self.decode_vehicle_type2(1, order_machines, route_machines, next_open_bale[:])
                self.update_openset(next_open_bale, next_open_pick, route_infos, next_open_bale_based)
                meta_infos["route_baler"] = route_infos
            elif mtype == "Picker" or mtype == "Truck"  :
                update_open_closed_time(self.dataset_lookups, "Picker", route_infos, isDebug)
                is_complete, route_infos = self.decode_vehicle_type2(2, order_machines, route_machines, next_open_pick[:])
                #self.update_openset(next_open_bale, next_open_pick, route_infos, next_open_bale_based)
                meta_infos["route_picker"] = route_infos
            
            for routeinfo in route_infos:
                print(routeinfo['route'])

    
    def decode_vehicle_type2(self, machie_id_type, order_machines, route_machines, open_set,  isOffset=False):
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        Start_Times = self.dataset_lookups["Start_Time"]
        End_Times = self.dataset_lookups["End_Time"]
        Setup_Times = self.dataset_lookups["Setup_Time"]
        Areas = self.dataset_lookups["Area"] #TODO
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]
        NF = self.dataset_lookups["Number_Feilds_NF"]
        Machine_Types = self.dataset_lookups["Machine_Type"]
        time_ids = self.dataset_lookups["Time_ID"]
        PRIORITY_FEILDS = self.dataset_lookups["PRIORITY_FEILDS"]
        ROUTES = self.dataset_lookups["ROUTE_ID"]
        
        CLOSED_TIMES = self.dataset_lookups["Closed_Time"]
        MACHINE_ID_TYPE_LOOKUP = self.dataset_lookups["MACHINE_ID_TYPE_LOOKUP"]
        MACHINE_NAME_TYPE_LOOKUP = self.dataset_lookups["MACHINE_NAME_TYPE_LOOKUP"]
        machine_name_type = MACHINE_NAME_TYPE_LOOKUP[machie_id_type]

        first_ids = np.argsort(x[:NF])
        ids = np.argsort(x[NF:])
        #print("first_ids", first_ids, time_ids[first_ids])
        #print("sort_index", ids)
        
        machines = list(np.where(Machine_Types == machine_name_type)[0])
        #print(seed)
        #r1 =np.random.default_rng(int(seed*1000))
        #fidx= int(r1.random()*len(first_ids))
        #fid = time_ids[first_ids[fidx]]
        #np.arange(NF)
        #print(fid, open_set)
        #while fid not in open_set:
            #fidx= int(r1.random()*len(first_ids))
            #fid = time_ids[first_ids[fidx]]
        
        #open_set = list(ids[:])
        route_infos = []
        close_set = []
        #print("open_times", open_times)
        #print("sort_index", ids)
        is_complete = True
        for m in range(len(order_machines)):
            machine = order_machines[m]
            route = route_machines[m]
            open_day_times =  np.copy(self.dataset_lookups["OPEN_TYPE_TIMES"][machine_name_type] )
            closed_day_times = np.copy( self.dataset_lookups["CLOSED_TYPE_TIMES"][machine_name_type])
            offset_times = np.copy(self.dataset_lookups["OFSSET_TIMES"][machine_name_type])
            machine_rate = Operation_Rates[machine]
            machine_working_time = End_Times[machine] - Start_Times[machine] -1 #remove break time
            
            idx = list(np.where(offset_times >= machine_working_time*60*0.75)[0])
            open_day_times[idx] += 1
            offset_times[idx] = 0

            open_times =  open_day_times*machine_working_time*60 + offset_times
            closed_times = closed_day_times*machine_working_time*60
            if machine >=  1000:
                print(f"Machine{machine} {Machine_Types}")
                print("open_times", open_times)
                print("closed_times", closed_times)
            isFound = False
            isCompleted, new_route_info= self.create_route_info(machine, machie_id_type, route, open_times, closed_times, isOffset)
            print("isCompleted", isCompleted)
            #print("new_route_info", new_route_info)
            is_complete = isCompleted and is_complete
            route_infos.append(new_route_info)
        return is_complete, route_infos

def display_result(rinfos):
    print("----------------------------------------------")
    for rinfo in rinfos:
        print("M{0}".format(rinfo['machine']+1), rinfo['st'])
        print("M{0}".format(rinfo['machine']+1), rinfo['ot'])
        print("M{0}".format(rinfo['machine']+1), rinfo['et'])
        print("M{0}".format(rinfo['machine']+1), rinfo['days'])
        print("M{0}".format(rinfo['machine']+1), rinfo['delta_day'])
        print("M{0}".format(rinfo['machine']+1), rinfo['route'])
    print()
        
def display_result_all(rinfoss):
    names = ["Sweeper", "Baler", "Picker", "Truck"]
    for i in range(len(rinfoss)):
        print(names[i])
        display_result(rinfoss[i])
        
if __name__ == "__main__":
    user_id = "U16"
    data_set = "D1"
    status = []
    
    #dataset_lookups = get_dataset(user_id, data_set, datetime(2023, 4, 2), priority_FIDs=["97", "F104"])
    dataset_lookups = creaet_data_lookup_test("data.json")
    decoder = Decoder(dataset_lookups)
    N = dataset_lookups["Number_Feilds"]
    NF = dataset_lookups["Number_Feilds_NF"]
    np.random.seed(0)
    x = np.random.rand(4+ 4*(N+NF))
    meta_infos =decoder.decode(x, True)
    #meta_infos
    rinfos1, rinfos2, rinfos3, rinfos4 = meta_infos['route_sweeper'], meta_infos['route_baler'], meta_infos['route_picker'], meta_infos[ 'route_truck']
    
    #display_result_all([rinfos1])
    #display_result_all([rinfos1, rinfos2, rinfos3, rinfos4])
    print("NF", NF, len(rinfos1[0]['route']))
    #print(meta_infos["cost_data"])
    routes = dataset_lookups["ROUTE"]
    decoder.regenerate_route_infos(routes)




#['F100' 'F101' 'F102' 'F103' 'F104' 'F105' 'F96' 'F97' 'F98' 'F99']
#['M41' 'M42' 'M43' 'M44']
