import requests as req
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta 


class Decoder:
    def __init__(self, dataset_lookups):
        self.dataset_lookups = dataset_lookups

    def getNext(self, start, ids, closedset, openset, ct, current_id, days, max_time,
                machine_working_time,machine_ready_times, machine_closed_times, machine_rate, machine_setup_time):
        ci = start
        M = len(ids)
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]
        Areas = self.dataset_lookups["Area"]
        while ci < M:
            cid = ids[ci]
            ci += 1
            if cid in closedset:
                continue

            next_t = ct + DM[current_id][cid]/TRAVELING_SPEED
            if next_t > (days + 1)*machine_working_time:
                next_t = (days + 1)*machine_working_time + DM[current_id][cid]/TRAVELING_SPEED
            #waiting to much times
            if  machine_ready_times[cid] - next_t  > machine_working_time*MAX_WAITING_DAYS:
                #swap 
                print("1 manage-----MAX_WAITING_DAYS-------------------------------", next_t, machine_ready_times[cid], machine_closed_times[cid])
                openset.append(cid)
                continue
            if next_t > machine_closed_times[cid]:
                print("2 manage----machine_closed_times--------------------------------")
                openset.append(cid)
                continue
            ot = round(Areas[cid]/machine_rate + machine_setup_time, 2) 
            if next_t + ot  > max_time:
                print("3 manage---max_time---------------------------------")
                openset.append(cid)
                continue
            break
        if ci == len(ids) and (cid in openset):
            return -1, -1
        return cid, ci

    def create_route(self, machine, fid, close_set, ids,  open_times, closed_times, isOffset=False ):
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        Start_Times = self.dataset_lookups["Start_Time"]
        End_Times = self.dataset_lookups["End_Time"]
        Setup_Times = self.dataset_lookups["Setup_Time"]
        Areas = self.dataset_lookups["Area"]
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]

        machine_rate = Operation_Rates[machine]
        machine_working_time = End_Times[machine] - Start_Times[machine] -1
        machine_setup_time = Setup_Times[machine]

        if isOffset:
            days = (open_times//24)
            machine_ready_times = days*machine_working_time + np.maximum((open_times - ((days*24) + Start_Times[machine])), 0)
            #print("xxxx", days*machine_working_time)
            #print("xxxx", np.maximum((open_times - ((days*24) + Start_Times[machine])), 0))
        
            #print("machine_ready_times x ", machine_working_time,days, machine_working_time, machine_ready_times)
            #print("open_times",open_times - ((days*24) + Start_Times[machine]))
        else:
            machine_ready_times = open_times/24*machine_working_time
        machine_closed_times = closed_times/24*machine_working_time
        max_time = np.max(machine_closed_times)
        #print("machine_ready_times", machine_ready_times)
        #print("machine_closed_times", machine_closed_times)

        ct = 0
        days = 0
        #print("current day", days)
        machine_ready_time = max(ct, days *machine_working_time )
        st = max(machine_ready_times[fid], machine_ready_time)
        ot = round(Areas[fid]/machine_rate+ machine_setup_time, 2) 
        et = st + ot
        days = et//machine_working_time
        #print("new day", days)
        ct = et
        

        openset = []
        current_id = fid
        route = [current_id]
        close_set.append(current_id)
        dts = [0]
        sts = [st ]
        ots = [ot]
        ets = [et]
        dss = [0]
        
        #print("=============== 1")
        #print("route", route)
        #print("machine rate", machine_rate)
        #print("max_time", max_time)
        #print("working time of machine", machine_working_time)
        #print("travel times", dts)
        #print("start times", sts)
        #print("operation times", ots)
        #print("end times", ets)
        #print("closedset", closedset)
        #print("openset", openset)
        #print()

        M = len(ids)
        ci = 0
        while ci < M:
            #print("========================= x", ci, ct)
            cid, ci = self.getNext(ci, ids, close_set, openset, ct, current_id, days, max_time,
                    machine_working_time,machine_ready_times, machine_closed_times, machine_rate, machine_setup_time)
            #print("========================= x", cid, ci)
            if cid in close_set:
                ci+=1
                continue

            if cid == -1:
                break
            
            dt = round(DM[current_id][cid]/TRAVELING_SPEED, 2)
            ds = DM[current_id][cid]
            machine_ready_time = dt + et
            next_days =  machine_ready_time // machine_working_time 
            if next_days != days:
                #print("Days not same", next_days, days, machine_ready_time, et)
                machine_ready_time = next_days*machine_working_time+dt
            
            if machine_ready_times[cid]//machine_working_time > machine_ready_time//machine_working_time:
                machine_ready_time = next_days*machine_working_time
            #print("next", cid, machine_ready_times[cid], machine_closed_times[cid])
            st = round(max(machine_ready_times[cid], machine_ready_time), 2)
            ot = round(Areas[cid]/machine_rate + machine_setup_time, 2) 
            et = round(st+ot, 2)
            if et > max_time:
                #print("Overtime")
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
            days = ct//machine_working_time
            #ci += 1

        

            
            print("route", route)
            print("machine rate", machine_rate)
            print("working time of machine", machine_working_time)
            print("travel times", dts)
            print("start times", sts)
            print("operation times", ots)
            print("end times", ets)
            print("closedset", closedset)
            print("openset", openset)
            print()
        
        route_info ={"machine": machine,
                    "route": route, 
                    "dt": dts,
                    "st": sts,
                    "ot": ots,
                    "et": ets,
                    "ds": dss,
                    }

        return route_info, close_set, openset


    def create_route_info(self, machine, route, open_times, closed_times, isOffset=False):
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        Start_Times = self.dataset_lookups["Start_Time"]
        End_Times = self.dataset_lookups["End_Time"]
        Setup_Times = self.dataset_lookups["Setup_Time"]
        Areas = self.dataset_lookups["Area"]
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]


        machine_rate = Operation_Rates[machine]
        machine_working_time = End_Times[machine] - Start_Times[machine] -1

        if isOffset:
            days = (open_times//24)
            machine_ready_times = days*machine_working_time + np.maximum((open_times - ((days*24) + Start_Times[machine])), 0)
        else:
            machine_ready_times = open_times/24*machine_working_time

        machine_closed_times = closed_times/24*machine_working_time
        max_time = np.max(machine_closed_times)
        machine_setup_time = Setup_Times[machine]

        #print("machine_ready_times_1", machine_ready_times, route)
        #print("machine_closed_times", machine_closed_times)

        ct = 0
        days = 0
        
        cid = route[0]
        current_id = cid
        machine_ready_time = max(ct, days *machine_working_time )
        st = max(machine_ready_times[current_id], machine_ready_time)
        ot = round(Areas[cid]/machine_rate + machine_setup_time, 2) 
        et = st + ot
        days = et//machine_working_time
        #print("new day", days)
        ct = et

        cts = [0]
        sts = [st]
        dts = [0]
        ots = [ot]
        ets = [et]
        dss = [0]
        isError = False
        #print("new day cid{0} st{1}, ot{2}, et{3}".format(cid, st, ot, et))
        for i in range(1, len(route)):
            cid = route[i]
            dt = round(DM[current_id][cid]/TRAVELING_SPEED, 2)
            ds = DM[current_id][cid]
            machine_ready_time = dt + et
            next_days =  machine_ready_time // machine_working_time 
            if next_days != days:
                #print("Days not same", next_days, days, machine_ready_time, et)
                machine_ready_time = next_days*machine_working_time+dt
            
            if machine_ready_times[cid]//machine_working_time > machine_ready_time//machine_working_time:
                machine_ready_time = next_days*machine_working_time
            #print("AAAAA next", cid, et,dt, next_days, machine_ready_time, machine_ready_times[cid], machine_closed_times[cid])
            st = round(max(machine_ready_times[cid], machine_ready_time), 2)
            ot = round(Areas[cid]/machine_rate + machine_setup_time, 2) 
            et = round(st+ot, 2)

            if et > machine_closed_times[cid]:
                isError = True
                #print("Error", cid, et, machine_closed_times[cid])
                break

            if et > max_time:
                #print("Overtime")
                #print("Error max", et, machine_closed_times[cid])
                break
            dts.append(dt)
            sts.append(st)
            ots.append(ot)
            ets.append(et)
            dss.append(ds)
            current_id = cid
            ct = et
            days = ct//machine_working_time
        
        route_info ={"machine": machine,
                    "route": route, 
                    "dt": dts,
                    "st": sts,
                    "ot": ots,
                    "et": ets,
                    "ds":dss
                    }

        if et > max_time:
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



    def decode_vehicle_type(self, name_type, x, open_set, open_times, closed_times, isOffset=False, seed=0):
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
        OPEN_TIMES = self.dataset_lookups["Open_Time"]
        CLOSED_TIMES = self.dataset_lookups["Closed_Time"]

        first_ids = np.argsort(x[:NF])
        ids = np.argsort(x[NF:])
        #print("first_ids", first_ids, time_ids[first_ids])
        #print("sort_index", ids)

        machines = list(np.where(Machine_Types == name_type)[0])
        #print(seed)
        r1 =np.random.default_rng(int(seed*1000))
        fidx= int(r1.random()*len(first_ids))
        fid = time_ids[first_ids[fidx]]
        #print(fid, open_set)
        while fid not in open_set:
            fidx= int(r1.random()*len(first_ids))
            fid = time_ids[first_ids[fidx]]
        
        #open_set = list(ids[:])
        route_infos = []
        close_set = []
        #print("open_times", open_times)
        for m in range(len(machines)):
            #print(" openset_0", open_set, close_set)
            machine = machines[m]
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
            open_set_subs = [fid]
            #print(" openset", open_set_subs, open_set, close_set)
            route_info, close_set, open_set_out  = self.create_route(machine, fid, close_set, open_set_subs, open_times, closed_times, isOffset)
            if len(open_set_out) == 0:
                open_set.remove(item)
                #
                #print("remove openset", item, open_set_subs, open_set, close_set)
            else:
                print("Errror---------------------")
            #print(" openset 3", open_set)
            if route_info == None:
                print("SHOLD NOT-------------------------------------------------")
            route_infos.append(route_info)
        # print("route_info", route_info)
            #print("open_set", open_set, )
        # print("close_set", close_set, )
            if len(open_set) == 0:
                break
        
        #print("last open_set", open_set, len(route_infos), route_infos[0])
        count_complete = 0
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
                for j in range(0, len(route)+1):
                    route = route_info['route'][:]
                    route.insert(j, cid)
                    #print("check route", route)
                    isCompleted, new_route_info= self.create_route_info(machine, route, open_times, closed_times, isOffset)
                    cost = np.sum(new_route_info['dt']) - 1000000*len(new_route_info['route'])
                    if isCompleted and bestCost > cost:
                        bestCost = cost
                        bestK = k
                        bestJ = j
                        bestRouteInfo = new_route_info
            if bestK == -1:
                continue
            if bestRouteInfo == None:
                continue
                
            route_infos[bestK] = bestRouteInfo
            count_complete += 1
            
            #print(cid, bestRouteInfo['route'])
        #print("---------------------", route_infos)
        if count_complete != len(open_set):
            return False, route_infos
        return True, route_infos
        


    def decode(self, x):
        ##print("=====================================================")
        #print("เครื่องกวาด")
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
        OPEN_TIMES = self.dataset_lookups["Open_Time"]
        CLOSED_TIMES = self.dataset_lookups["Closed_Time"]
        meta_infos = {}
        N = self.dataset_lookups["Number_Feilds"]

        next_open = []
        for i in range(N):
            if self.dataset_lookups["Is_Sweep"][i] == 1:
                continue
            next_open.append(i)

        #[['Sweeper'], ['Baler'], ['Baler'], ['Picker'], ['Truck']]
        is_complete, route_infos = self.decode_vehicle_type('Sweeper',x[:N+NF], next_open[:], OPEN_TIMES, CLOSED_TIMES, seed=x[4*(N+NF)])
        last_open = next_open
        next_open = []
        if not is_complete:
            #print(route_infos[0])
            for r in range(len(route_infos)):
                next_open.extend(route_infos[r]['route'])
            #print("next_open", next_open)
        else:
            for i in last_open:
                next_open.append(i)

        for i in range(N):
            #print(i)
            if self.dataset_lookups["Is_Baled"][i] == 0:
                if i not in next_open:
                    next_open.append(i)
            else:
                if i in next_open:
                    next_open.remove(i)
        #print(next_open)

        total = 0
        second_closed_times = CLOSED_TIMES + 5*24
        second_open_times= np.copy(OPEN_TIMES)
        #print("closed_times", CLOSED_TIMES )
        #print("second_closed_times", second_closed_times )
        for k in range(len(route_infos)):
            route_info = route_infos[k]
            total += len(route_info['route'][:])
            route = route_info['route']
            machine = route_info['machine']
            st = np.array(route_info['st'])
            et = np.array(route_info['et'])
            machine_rate = Operation_Rates[machine]
            machine_working_time = End_Times[machine] - Start_Times[machine] -1
            #print("machine", machine, machine_rate, machine_working_time)
            for j in range(len(route)):
                #j = route[jj]
                day = et[j]//machine_working_time
                #print("day", day)
                delta = round(et[j] - day*machine_working_time, 2)
                hour = round(day*24+ delta + Start_Times[machine], 2)
                #print(machine, j, route[j], et[j], day, hour//24, delta, hour, Start_Times[machine])
                second_open_times[route[j]] = hour
        #print("Total", total)
        ##print("open_times", OPEN_TIMES )
        #print("second_open_times", second_open_times)
        #print("=====================================================")
        #print()
        
        #print("=====================================================")
        #print("เครื่องคีบ")
        #print("second_open_times", second_open_times)
        #print("second_closed_times", second_closed_times)
        is_complete, route_infos2 = self.decode_vehicle_type('Baler',x[(N+NF):(N+NF)*2], next_open[:], second_open_times, second_closed_times, isOffset=True , seed=x[4*(N+NF)+1])
        last_open = next_open
        next_open = []
        if not is_complete:
            #print(route_infos[0])
            for r in range(len(route_infos2)):
                next_open.extend(route_infos2[r]['route'])
            print("next_open",next_open)
        else:
            for i in last_open:
                next_open.append(i)

        for i in range(N):
            if self.dataset_lookups["Is_Picked"][i] == 0:
                if i not in next_open:
                    next_open.append(i)
            else:
                if i in next_open:
                    next_open.remove(i)
        
        total = 0
        thrid_closed_times = second_closed_times
        thrid_open_times= np.copy(OPEN_TIMES)
        #print("thrid_closed_times", thrid_closed_times )
        for k in range(len(route_infos2)):
            route_info = route_infos2[k]
            total += len(route_info['route'][:])
            route = route_info['route']
            machine = route_info['machine']
            st = np.array(route_info['st'])
            et = np.array(route_info['et'])
            #print("route", route)
            #print("st", st)
            #print("et", et)
            machine_rate = Operation_Rates[machine]
            machine_working_time = End_Times[machine] - Start_Times[machine] -1
            #print("machine", machine, machine_rate, machine_working_time)
            for j in range(len(route)):
                day = et[j]//machine_working_time
                delta = round(et[j] - day*machine_working_time, 2)
                hour = round(day*24+ delta + Start_Times[machine], 2)
                #print(machine, j, route[j], et[j], day, hour//24, delta, hour)
                thrid_open_times[route[j]] =  hour 
        #print("Total", total)
        #print("second_open_times", second_open_times )
        #print("thrid_open_times", thrid_open_times)
        #print("=====================================================")
        #print()
        
        #print("=====================================================")
        #print("เครื่องอัด")
        #print("thrid_open_times", thrid_open_times )
        #print("thrid_closed_times", thrid_closed_times - thrid_open_times)
        is_complete, route_infos3 = self.decode_vehicle_type('Picker',x[2*(N+NF):3*(N+NF)], next_open[:], thrid_open_times, thrid_closed_times, isOffset=True, seed=x[4*(N+NF)+2])
        #print("route_infos3", route_infos3)
        last_open = next_open
        next_open = []
        if not is_complete:
            #print(route_infos[0])
            for r in range(len(route_infos3)):
                next_open.extend(route_infos3[r]['route'])
            ##print(next_open)
        else:
            for i in last_open:
                next_open.append(i)
            #print("next_open", next_open)
        total = 0
        forth_open_times= np.copy(OPEN_TIMES)
        
        for k in range(len(route_infos3)):
            route_info = route_infos3[k]
            if route_info == None:
                print("-------------------------------------------------------------------", route_infos3)
            total += len(route_info['route'][:])
            route = route_info['route']
            machine = route_info['machine']
            st = np.array(route_info['st'])
            et = np.array(route_info['et'])
            machine_rate = Operation_Rates[machine]
            machine_working_time = End_Times[machine] - Start_Times[machine] -1
            #print("machine", machine, machine_rate, machine_working_time)
        # print(route)
        # print(et)
            for j in range(len(route)):
                day = et[j]//machine_working_time
                delta = round(et[j] - day*machine_working_time, 2)
                hour = round(day*24+ delta + Start_Times[machine], 2)
                #print(machine, j, route[j], et[j], day, hour//24, delta, hour)
                forth_open_times[route[j]] = hour
        
        meta_infos["Open time second"] = second_open_times  
        meta_infos["Closed time second"] = second_closed_times 
        meta_infos["Open time thrid"] = thrid_open_times 
        meta_infos["Closed time thrid"] = thrid_closed_times 
        #return route_infos, route_infos2, route_infos3
    

        return route_infos, route_infos2, route_infos3,route_infos3, meta_infos


class Decoder2:
    def __init__(self, dataset_lookups):
        self.dataset_lookups = dataset_lookups

    def getNext(self, start, ids, closedset, openset, ct, current_id, days, max_time,
                machine_working_time,machine_ready_times, machine_closed_times, machine_rate, machine_setup_time):
        ci = start
        M = len(ids)
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]
        Areas = self.dataset_lookups["Area"]
        while ci < M:
            cid = ids[ci]
            ci += 1
            if cid in closedset:
                continue

            next_t = ct + DM[current_id][cid]/TRAVELING_SPEED
            if next_t > (days + 1)*machine_working_time:
                next_t = (days + 1)*machine_working_time + DM[current_id][cid]/TRAVELING_SPEED
            #waiting to much times
            if  machine_ready_times[cid] - next_t  > machine_working_time*MAX_WAITING_DAYS:
                #swap 
                print("1 manage-----MAX_WAITING_DAYS-------------------------------", next_t, machine_ready_times[cid], machine_closed_times[cid])
                openset.append(cid)
                continue
            if next_t > machine_closed_times[cid]:
                print("2 manage----machine_closed_times--------------------------------")
                openset.append(cid)
                continue
            ot = round(Areas[cid]/machine_rate + machine_setup_time, 2) 
            if next_t + ot  > max_time:
                print("3 manage---max_time---------------------------------")
                openset.append(cid)
                continue
            break
        if ci == len(ids) and (cid in openset):
            return -1, -1
        return cid, ci

    def create_route(self, machine, fid, close_set, ids,  open_times, closed_times, isOffset=False ):
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        Start_Times = self.dataset_lookups["Start_Time"]
        End_Times = self.dataset_lookups["End_Time"]
        Setup_Times = self.dataset_lookups["Setup_Time"]
        Areas = self.dataset_lookups["Area"]
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]

        machine_rate = Operation_Rates[machine]
        machine_working_time = End_Times[machine] - Start_Times[machine] -1
        machine_setup_time = Setup_Times[machine]

        if isOffset:
            days = (open_times//24)
            machine_ready_times = days*machine_working_time + np.maximum((open_times - ((days*24) + Start_Times[machine])), 0)
            #print("xxxx", days*machine_working_time)
            #print("xxxx", np.maximum((open_times - ((days*24) + Start_Times[machine])), 0))
        
            #print("machine_ready_times x ", machine_working_time,days, machine_working_time, machine_ready_times)
            #print("open_times",open_times - ((days*24) + Start_Times[machine]))
        else:
            machine_ready_times = open_times/24*machine_working_time
        machine_closed_times = closed_times/24*machine_working_time
        max_time = np.max(machine_closed_times)
        #print("machine_ready_times", machine_ready_times)
        #print("machine_closed_times", machine_closed_times)

        ct = 0
        days = 0
        #print("current day", days)
        machine_ready_time = max(ct, days *machine_working_time )
        st = max(machine_ready_times[fid], machine_ready_time)
        ot = round(Areas[fid]/machine_rate+ machine_setup_time, 2) 
        et = st + ot
        days = et//machine_working_time
        #print("new day", days)
        ct = et
        

        openset = []
        current_id = fid
        route = [current_id]
        close_set.append(current_id)
        dts = [0]
        sts = [st ]
        ots = [ot]
        ets = [et]
        dss = [0]
        
        #print("=============== 1")
        #print("route", route)
        #print("machine rate", machine_rate)
        #print("max_time", max_time)
        #print("working time of machine", machine_working_time)
        #print("travel times", dts)
        #print("start times", sts)
        #print("operation times", ots)
        #print("end times", ets)
        #print("closedset", closedset)
        #print("openset", openset)
        #print()

        M = len(ids)
        ci = 0
        while ci < M:
            #print("========================= x", ci, ct)
            cid, ci = self.getNext(ci, ids, close_set, openset, ct, current_id, days, max_time,
                    machine_working_time,machine_ready_times, machine_closed_times, machine_rate, machine_setup_time)
            #print("========================= x", cid, ci)
            if cid in close_set:
                ci+=1
                continue

            if cid == -1:
                break
            
            dt = round(DM[current_id][cid]/TRAVELING_SPEED, 2)
            ds = DM[current_id][cid]
            machine_ready_time = dt + et
            next_days =  machine_ready_time // machine_working_time 
            if next_days != days:
                #print("Days not same", next_days, days, machine_ready_time, et)
                machine_ready_time = next_days*machine_working_time+dt
            
            if machine_ready_times[cid]//machine_working_time > machine_ready_time//machine_working_time:
                machine_ready_time = next_days*machine_working_time
            #print("next", cid, machine_ready_times[cid], machine_closed_times[cid])
            st = round(max(machine_ready_times[cid], machine_ready_time), 2)
            ot = round(Areas[cid]/machine_rate + machine_setup_time, 2) 
            et = round(st+ot, 2)
            if et > max_time:
                #print("Overtime")
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
            days = ct//machine_working_time
            #ci += 1

        

            
            print("route", route)
            print("machine rate", machine_rate)
            print("working time of machine", machine_working_time)
            print("travel times", dts)
            print("start times", sts)
            print("operation times", ots)
            print("end times", ets)
            print("closedset", closedset)
            print("openset", openset)
            print()
        
        route_info ={"machine": machine,
                    "route": route, 
                    "dt": dts,
                    "st": sts,
                    "ot": ots,
                    "et": ets,
                    "ds": dss,
                    }

        return route_info, close_set, openset


    def create_route_info(self, machine, route, open_times, closed_times, isOffset=False):
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        Start_Times = self.dataset_lookups["Start_Time"]
        End_Times = self.dataset_lookups["End_Time"]
        Setup_Times = self.dataset_lookups["Setup_Time"]
        Areas = self.dataset_lookups["Area"]
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]


        machine_rate = Operation_Rates[machine]
        machine_working_time = End_Times[machine] - Start_Times[machine] -1

        if isOffset:
            days = (open_times//24)
            machine_ready_times = days*machine_working_time + np.maximum((open_times - ((days*24) + Start_Times[machine])), 0)
        else:
            machine_ready_times = open_times/24*machine_working_time

        machine_closed_times = closed_times/24*machine_working_time
        max_time = np.max(machine_closed_times)
        machine_setup_time = Setup_Times[machine]

        #print("machine_ready_times_1", machine_ready_times, route)
        #print("machine_closed_times", machine_closed_times)

        ct = 0
        days = 0
        
        cid = route[0]
        current_id = cid
        machine_ready_time = max(ct, days *machine_working_time )
        st = max(machine_ready_times[current_id], machine_ready_time)
        ot = round(Areas[cid]/machine_rate + machine_setup_time, 2) 
        et = st + ot
        days = et//machine_working_time
        #print("new day", days)
        ct = et

        cts = [0]
        sts = [st]
        dts = [0]
        ots = [ot]
        ets = [et]
        dss = [0]
        isError = False
        #print("new day cid{0} st{1}, ot{2}, et{3}".format(cid, st, ot, et))
        for i in range(1, len(route)):
            cid = route[i]
            dt = round(DM[current_id][cid]/TRAVELING_SPEED, 2)
            ds = DM[current_id][cid]
            machine_ready_time = dt + et
            next_days =  machine_ready_time // machine_working_time 
            if next_days != days:
                #print("Days not same", next_days, days, machine_ready_time, et)
                machine_ready_time = next_days*machine_working_time+dt
            
            if machine_ready_times[cid]//machine_working_time > machine_ready_time//machine_working_time:
                machine_ready_time = next_days*machine_working_time
            #print("AAAAA next", cid, et,dt, next_days, machine_ready_time, machine_ready_times[cid], machine_closed_times[cid])
            st = round(max(machine_ready_times[cid], machine_ready_time), 2)
            ot = round(Areas[cid]/machine_rate + machine_setup_time, 2) 
            et = round(st+ot, 2)

            if et > machine_closed_times[cid]:
                isError = True
                #print("Error", cid, et, machine_closed_times[cid])
                break

            if et > max_time:
                #print("Overtime")
                #print("Error max", et, machine_closed_times[cid])
                break
            dts.append(dt)
            sts.append(st)
            ots.append(ot)
            ets.append(et)
            dss.append(ds)
            current_id = cid
            ct = et
            days = ct//machine_working_time
        
        route_info ={"machine": machine,
                    "route": route, 
                    "dt": dts,
                    "st": sts,
                    "ot": ots,
                    "et": ets,
                    "ds":dss
                    }

        if et > max_time:
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



    def decode_vehicle_type(self, name_type, x, open_set, open_times, closed_times, isOffset=False, seed=0):
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
        OPEN_TIMES = self.dataset_lookups["Open_Time"]
        CLOSED_TIMES = self.dataset_lookups["Closed_Time"]

        first_ids = np.argsort(x[:NF])
        ids = np.argsort(x[NF:])
        #print("first_ids", first_ids, time_ids[first_ids])
        #print("sort_index", ids)

        machines = list(np.where(Machine_Types == name_type)[0])
        #print(seed)
        r1 =np.random.default_rng(int(seed*1000))
        fidx= int(r1.random()*len(first_ids))
        fid = time_ids[first_ids[fidx]]
        while fid not in open_set:
            fidx= int(r1.random()*len(first_ids))
            fid = time_ids[first_ids[fidx]]
        
        #open_set = list(ids[:])
        route_infos = []
        close_set = []
        #print("open_times", open_times)
        for m in range(len(machines)):
            #print(" openset_0", open_set, close_set)
            machine = machines[m]
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
            open_set_subs = [fid]
            #print(" openset", open_set_subs, open_set, close_set)
            route_info, close_set, open_set_out  = self.create_route(machine, fid, close_set, open_set_subs, open_times, closed_times, isOffset)
            if len(open_set_out) == 0:
                open_set.remove(item)
                #
                #print("remove openset", item, open_set_subs, open_set, close_set)
            else:
                print("Errror---------------------")
            #print(" openset 3", open_set)
            if route_info == None:
                print("SHOLD NOT-------------------------------------------------")
            route_infos.append(route_info)
        # print("route_info", route_info)
            #print("open_set", open_set, )
        # print("close_set", close_set, )
            if len(open_set) == 0:
                break
        
        #print("last open_set", open_set, len(route_infos), route_infos[0])
        count_complete = 0
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
                for j in range(0, len(route)+1):
                    route = route_info['route'][:]
                    route.insert(j, cid)
                    #print("check route", route)
                    isCompleted, new_route_info= self.create_route_info(machine, route, open_times, closed_times, isOffset)
                    cost = np.sum(new_route_info['dt']) - 1000000*len(new_route_info['route'])
                    if isCompleted and bestCost > cost:
                        bestCost = cost
                        bestK = k
                        bestJ = j
                        bestRouteInfo = new_route_info
            if bestK == -1:
                continue
            if bestRouteInfo == None:
                continue
                
            route_infos[bestK] = bestRouteInfo
            count_complete += 1
            
            #print(cid, bestRouteInfo['route'])
        #print("---------------------", route_infos)
        if count_complete != len(open_set):
            return False, route_infos
        return True, route_infos
        


    def decode(self, x):
        ##print("=====================================================")
        #print("เครื่องกวาด")
        Operation_Rates = self.dataset_lookups["Operation_Rate"]
        Start_Times = self.dataset_lookups["Start_Time"]
        End_Times = self.dataset_lookups["End_Time"]
        Setup_Times = self.dataset_lookups["Setup_Time"]
        Areas = self.dataset_lookups["Area"]
        DM = self.dataset_lookups["DM"]
        TRAVELING_SPEED = self.dataset_lookups["Traveling_Speed"]
        MAX_WAITING_DAYS = self.dataset_lookups["Max_Waiting_Days"]
        NF = self.dataset_lookups["Number_Feilds_NF"]
        N = self.dataset_lookups["Number_Feilds"]
        Machine_Types = self.dataset_lookups["Machine_Type"]
        time_ids = self.dataset_lookups["Time_ID"]
        OPEN_TIMES = self.dataset_lookups["Open_Time"]
        CLOSED_TIMES = self.dataset_lookups["Closed_Time"]
        meta_infos = {}

        next_open = []
        for i in range(N):
            next_open.append(i)

        #[['Sweeper'], ['Baler'], ['Baler'], ['Picker'], ['Truck']]
        is_complete, route_infos = self.decode_vehicle_type('Sweeper',x[:N+NF], next_open[:], OPEN_TIMES, CLOSED_TIMES, seed=x[4*(N+NF)])
        last_open = next_open
        next_open = []
        if not is_complete:
            #print(route_infos[0])
            for r in range(len(route_infos)):
                next_open.extend(route_infos[r]['route'])
            #print("next_open", next_open)
        else:
            for i in last_open:
                next_open.append(i)

        total = 0
        second_closed_times = CLOSED_TIMES + 5*24
        second_open_times= np.copy(OPEN_TIMES)
        #print("closed_times", CLOSED_TIMES )
        #print("second_closed_times", second_closed_times )
        for k in range(len(route_infos)):
            route_info = route_infos[k]
            total += len(route_info['route'][:])
            route = route_info['route']
            machine = route_info['machine']
            st = np.array(route_info['st'])
            et = np.array(route_info['et'])
            machine_rate = Operation_Rates[machine]
            machine_working_time = End_Times[machine] - Start_Times[machine] -1
            #print("machine", machine, machine_rate, machine_working_time)
            for j in range(len(route)):
                #j = route[jj]
                day = et[j]//machine_working_time
                #print("day", day)
                delta = round(et[j] - day*machine_working_time, 2)
                hour = round(day*24+ delta + Start_Times[machine], 2)
                #print(machine, j, route[j], et[j], day, hour//24, delta, hour, Start_Times[machine])
                second_open_times[route[j]] = hour
        #print("Total", total)
        ##print("open_times", OPEN_TIMES )
        #print("second_open_times", second_open_times)
        #print("=====================================================")
        #print()
        
        #print("=====================================================")
        #print("เครื่องคีบ")
        #print("second_open_times", second_open_times)
        #print("second_closed_times", second_closed_times)
        is_complete, route_infos2 = self.decode_vehicle_type('Baler',x[(N+NF):(N+NF)*2], next_open[:], second_open_times, second_closed_times, isOffset=True , seed=x[4*(N+NF)+1])
        last_open = next_open
        next_open = []
        if not is_complete:
            #print(route_infos[0])
            for r in range(len(route_infos2)):
                next_open.extend(route_infos2[r]['route'])
            print("next_open",next_open)
        else:
            for i in last_open:
                next_open.append(i)
        total = 0
        thrid_closed_times = second_closed_times
        thrid_open_times= np.copy(OPEN_TIMES)
        #print("thrid_closed_times", thrid_closed_times )
        for k in range(len(route_infos2)):
            route_info = route_infos2[k]
            total += len(route_info['route'][:])
            route = route_info['route']
            machine = route_info['machine']
            st = np.array(route_info['st'])
            et = np.array(route_info['et'])
            #print("route", route)
            #print("st", st)
            #print("et", et)
            machine_rate = Operation_Rates[machine]
            machine_working_time = End_Times[machine] - Start_Times[machine] -1
            #print("machine", machine, machine_rate, machine_working_time)
            for j in range(len(route)):
                day = et[j]//machine_working_time
                delta = round(et[j] - day*machine_working_time, 2)
                hour = round(day*24+ delta + Start_Times[machine], 2)
                #print(machine, j, route[j], et[j], day, hour//24, delta, hour)
                thrid_open_times[route[j]] =  hour 
        #print("Total", total)
        #print("second_open_times", second_open_times )
        #print("thrid_open_times", thrid_open_times)
        #print("=====================================================")
        #print()
        
        #print("=====================================================")
        #print("เครื่องอัด")
        #print("thrid_open_times", thrid_open_times )
        #print("thrid_closed_times", thrid_closed_times - thrid_open_times)
        is_complete, route_infos3 = self.decode_vehicle_type('Picker',x[2*(N+NF):3*(N+NF)], next_open[:], thrid_open_times, thrid_closed_times, isOffset=True, seed=x[4*(N+NF)+2])
        #print("route_infos3", route_infos3)
        last_open = next_open
        next_open = []
        if not is_complete:
            #print(route_infos[0])
            for r in range(len(route_infos3)):
                next_open.extend(route_infos3[r]['route'])
            ##print(next_open)
        else:
            for i in last_open:
                next_open.append(i)
            #print("next_open", next_open)
        total = 0
        forth_open_times= np.copy(OPEN_TIMES)
        
        for k in range(len(route_infos3)):
            route_info = route_infos3[k]
            if route_info == None:
                print("-------------------------------------------------------------------", route_infos3)
            total += len(route_info['route'][:])
            route = route_info['route']
            machine = route_info['machine']
            st = np.array(route_info['st'])
            et = np.array(route_info['et'])
            machine_rate = Operation_Rates[machine]
            machine_working_time = End_Times[machine] - Start_Times[machine] -1
            #print("machine", machine, machine_rate, machine_working_time)
        # print(route)
        # print(et)
            for j in range(len(route)):
                day = et[j]//machine_working_time
                delta = round(et[j] - day*machine_working_time, 2)
                hour = round(day*24+ delta + Start_Times[machine], 2)
                #print(machine, j, route[j], et[j], day, hour//24, delta, hour)
                forth_open_times[route[j]] = hour
        
        meta_infos["Open time second"] = second_open_times  
        meta_infos["Closed time second"] = second_closed_times 
        meta_infos["Open time thrid"] = thrid_open_times 
        meta_infos["Closed time thrid"] = thrid_closed_times 
        #return route_infos, route_infos2, route_infos3
    

        return route_infos, route_infos2, route_infos3,route_infos3, meta_infos

