import requests as req
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta 


def get_dataset(user_id, data_set):
    url = "https://www.otpzlab.com/api/data-get-where.php?"
    url1 = "https://green.manopna.online//api/data-get-where.php?"
    url2 ="https://www.otpzlab.com/DM%20FILE"
    url3 ="https://green.manopna.online/DM%20FILE"
    query_feild = '{0}db=field&where=user_id="{1}"%20and%20data_set="{2}"'.format(url1, user_id, data_set)
    
    query_machine = '{0}db=machines&where=user_id="{1}"%20and%20data_set="{2}"'.format(url1, user_id, data_set)
    
    try:
        query_dm = '{0}/{1}_{2}.csv'.format(url3, data_set, user_id)
        print(query_dm)
        dm_df = pd.read_csv(query_dm)
    except:
        query_dm = '{0}/{1}_{2}.csv'.format(url2, data_set, user_id)
        
        dm_df = pd.read_csv(query_dm)
    resp = req.get(query_feild)
    feilds = resp.json()
    #print(len(feilds))
    resp = req.get(query_machine)
    machines = resp.json()

    Status = []
    for i in range(len(feilds)):
        Status.append([0, 0, 0])

    for i in range(len(feilds)):
        if i < len(feilds):
            status = Status[i]
        else:
            status =[0, 0, 0]
        feilds[i]['is_sweep'] = status[0]
        feilds[i]['is_baled'] = status[1]
        feilds[i]['is_picked'] = status[2]

    #print(feilds)
    feild_df = pd.DataFrame(feilds)
    machine_df = pd.DataFrame(machines)

    #dm_df.to_csv("{0}_{1}.csv".format(data_set, user_id), encoding="utf-8", index=False)
    config_sweep_date = 3

    #print(len(machines), len(feilds))
    #feild_df

    LATs = feild_df['LAT'].to_numpy().astype(np.float64)
    LNGs = feild_df['LNG'].to_numpy().astype(np.float64)
    Names = feild_df['name'].to_numpy()
    Areas = feild_df['area_rai'].to_numpy().astype(np.float64)
    Cut_Dates = feild_df['cut_date'].to_numpy()
    Production_Rates = feild_df['production_rate_ton_per_rai'].to_numpy().astype(np.float64)
    IDs = feild_df['id'].to_numpy()
    Prices = np.ones(len(Production_Rates))*1000

    

    if "machine_type_id" in machine_df.columns:
        data = []
        lookup_names = {"1":"sweeper", "2":"baler", "3":"picker", "4":"truck"}
        for i in range(len(machine_df["machine_type_id"])):
            data.append(lookup_names[machine_df.iloc[i]["machine_type_id"]])
        machine_df['machine_type'] = data

    End_Times = machine_df['end_time'].to_numpy().astype(np.float64)
    Start_Times = machine_df['start_time'].to_numpy().astype(np.float64)
    Setup_Times = machine_df['setup_time'].to_numpy().astype(np.float64)
    Operation_Rates = machine_df['operation_rate_ton_hour'].to_numpy().astype(np.float64)
    Maintain_Costs = machine_df['maintain_cost_baht_hour'].to_numpy().astype(np.float64)
    Machine_Types = machine_df['machine_type'].to_numpy()
    Machine_IDs = machine_df['id'].to_numpy()
    Labor_Costs = machine_df['labor_cost'].to_numpy().astype(np.float64)
    Fuel_Rates = machine_df['fuel_rate_travel_liter_km'].to_numpy().astype(np.float64)

    print(query_machine)
    print(query_feild)
    print(query_dm)
    

    SPEED = 20
    DM = dm_df.to_numpy()[:, 1:]/1000
    print("DM", DM.shape)
    N = len(Cut_Dates)
    times = []
    for i in range(N):
        Cut_Dates[i] = datetime.datetime.strptime(Cut_Dates[i], '%Y-%m-%d')
    for i in range(len(Machine_Types)):
        Machine_Types[i] =  Machine_Types[i][0].upper() +  Machine_Types[i][1:]

    for i in range(len(DM)):
        DM[i][i] = 0
        for j in range(i):
            DM[i][j] = DM[j][i]

    DM[2][4] , DM[4][2]
    N = len(Cut_Dates)
    times = []
    startdate = datetime.datetime(2023, 1, 1)
    for i in range(N):
        dt = Cut_Dates[i] - startdate
        hour = dt.total_seconds()/60/60
        times.append(hour)
    OPEN_TIMES = np.array(times)
    OPEN_TIMES

    START_GROUP_RATIO = 1
    MAX_WAITING_DAYS = 2
    TRAVELING_SPEED = SPEED
    NF = int(N*START_GROUP_RATIO)
    time_ids = np.argsort(times) 
    CLOSED_TIMES = OPEN_TIMES + config_sweep_date*24
    print("Early Set", time_ids)
    print("Ready Time", OPEN_TIMES[time_ids])
    print("End Time", CLOSED_TIMES[time_ids])


    

    #print("End Time", Cut_Dates[time_ids])
    return { 
        "Lat":LATs,
        "Lng":LNGs,
        "Name":Names,
        "Area":Areas,
        "Cut_Date":Cut_Dates,
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
        "Labor_Cost":Labor_Costs,
        "Fuel_Rate":Fuel_Rates,
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
    }

def create_json(dataset_lookups, decoder, x):
    update_route_infos = []
    rinfos1, rinfos2, rinfos3,rinfos4, meta_infos =decoder.decode(x)
    rinfos = rinfos1[:]
    K2 = len(rinfos)
    rinfos.extend(rinfos2)
    K3 = len(rinfos)
    rinfos.extend(rinfos3)
    K4 = len(rinfos)
    rinfos.extend(rinfos4)
    K5 = len(rinfos)
    K = len(rinfos)

    Machine_IDs = dataset_lookups["Machine_ID"]
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
    DM = dataset_lookups["DM"]
    Names = dataset_lookups["Name"]


    types = []
    type_names = ["Sweeper", "Baler", "Picker", "Truck"]
    for k in range(len(type_names)):
        types.append([])

    for i in range(len(Machine_Types)):
        k = type_names.index(Machine_Types[i])
        types[k].append(i)

    print(types)



    for k in range(K):
        route_info = rinfos[k]
        route = rinfos[k]['route']
        machine = k
        start_work_machine = Start_Times[machine]
        end_work_machine = End_Times[machine]
        work_rate = Operation_Rates[machine]
        working_hour_machine = end_work_machine - start_work_machine - 1
        setup_time =  Setup_Times[machine]
        #print("Machine start working time:", start_work_machine)
        #print("Machine stop working time:", end_work_machine)
        #print("Machine working hour:", working_hour_machine)
        #print("Machine Working rate[rai/hour]:", work_rate)
        #print("Machine Setup time", setup_time)

        title_format = "{0}\t|{1}\t|{2}\t|{3}\t|{4}\t|{5}\t|{6}\t|{7}\t|{8}\t|{9}\t"
        #print(title_format.format("ID", "OPEN", "CLOSED", "Dist", "Area",  "Travel", "Arrive", "Start",  "OR", "Exit"))
        last_exit = 0

        update_route_info = []

        open_times = OPEN_TIMES
        closed_times = CLOSED_TIMES
        
        if machine >=K2 and machine < K3:
            open_times = meta_infos["Open time second"]
            closed_times = meta_infos["Closed time second"]
        if machine >= K3:
            open_times = meta_infos["Open time thrid"]
            closed_times = meta_infos["Closed time thrid"]

        #if machine > K2:
            #days = (open_times//24)
            #machine_ready_times = days*machine_working_time + np.maximum((open_times - ((days*24) + Start_Times[machine])), 0)
        #else:
            #machine_ready_times = open_times/24*machine_working_time

    
        #print("machine_ready_times", machine_ready_times)


        for i in range(len(route)):
            id = route[i]
            name = Names[id]
            
            open_time = open_times[id]
            close_time = closed_times[id]
            dt = route_info['dt'][i]
            st = route_info['st'][i]
            et = route_info['et'][i]
            ot = route_info['ot'][i]
            #print(dt, st, ot, et)
            #print(i , "i")
            if i == 0:
                km = 0
            else:
                km = DM[prev][id]
            area= Areas[id]
            #start_work = round((st//working_hour_machine)*24 + start_work_machine + st, 2)
            """
            day = st//working_hour_machine
            #day = max(day, open_time//24)
            #print("Day", day, st, working_hour_machine)
            nst = max(st, open_time//24*working_hour_machine)
            nday = nst//working_hour_machine
            print("Day", day, st, working_hour_machine)
            delta = round(nst - nday*working_hour_machine, 2)
            st_hour = round(day*24 + delta+ working_hour_machine, 2)
            #day = et//machine_working_time
            daye = et//working_hour_machine
            delta = round(et - daye*working_hour_machine, 2)
            et_hour = round(daye*24+ delta + Start_Times[machine], 2)
            arrive = round(last_exit + dt, 2)
            """
            day = st//working_hour_machine
            delta = round(st - day*working_hour_machine, 2)
            if delta < 0:
                delta = 0
            
            yst_hour = round(day*24+ delta + Start_Times[machine], 2)
            daye = et//working_hour_machine
            
            delta2 = round(et - daye*working_hour_machine, 2)
            yet_hour = round(daye*24+ delta2 + Start_Times[machine], 2)
            et_hour = round(daye*24+ delta2 + Start_Times[machine], 2)
            arrive = round(last_exit + dt, 2)
            #print(day,delta, dt, yst_hour, ot, yet_hour)
            #print(day,daye, st, et)
            if day != daye:
                #print("Done same day", day, daye)
                yet_hour = []
                yet_hour.append(round(daye*24 + Start_Times[machine]+ working_hour_machine, 2))
                yet_hour.append(round(daye*24 + Start_Times[machine], 2))
                yet_hour.append(round(daye*24 + Start_Times[machine] + delta, 2))
                last_exit = yet_hour[-1]
            else:
                last_exit = yet_hour
            #st_hour = st
            #et_hour = et

            #print(title_format.format(name, open_time, close_time, km, area,  dt, arrive, yst_hour,  ot,  yet_hour))
            #print(title_format.format(name, open_time, close_time, km, area,  dt, arrive, st,  ot,  et))
            #print(open_time, close_time, start_work_machine, )
            prev = id
            update_route_info.append({'machine':machine,  'machine type':Machine_Types[machine], "Farm ID":id,  'LAT':LATs[id], 'LNG':LNGs[id], "name": name, 'open':open_time, 'close':close_time, 'km':km,
                                    "area": area, 'travel time':dt, 'arrive time':arrive, 'start time':yst_hour, 
                                    'operation time': ot, 'end time': yet_hour})
            
        update_route_infos.append(update_route_info)

        
    N = len(Cut_Dates)
    times = []
    startdate = datetime.datetime(2023, 1, 1)
    df = []
    annots =[]
    last_end = min(OPEN_TIMES)
    M = len(update_route_infos)

    #types = [[0,1], [2, 3, 4], [5, 6, 7], [8, 9, 10]]
    #type_names = ["Sweeper", "Baler", "Picker", "Truck"]




    rows = []
    columns = [
                "solution",
                "machine_id",
                "farm_id",
                "lat",
                "lng",
                "machine_type",
                "open_date",
                "closed_date",
                "work_date",
                "distance",
                "travel_time",
                "start_time",
                "operation_time",
                "exit_time",
    ]


    temp = {
        "solution": "S1",
        "machine_id": "M1",
        "feild_id": "F1",
        "feild_name": "F1",
        "lat": LATs[0],
        "lng": LNGs[0],
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
    json_datas = []
    print("len(update_route_infos)", len(update_route_infos))

    for k in range(len(update_route_infos)):
        update_route_info = update_route_infos[k]
        machine = "Machine" + str(k+1)
        tye =-1

        for ty in range(4):
            if k in types[ty]:
                machine = type_names[ty] + str(k+1)
                tye = ty
                break

        machine_i= int(update_route_info[k]['machine'])
        #print(machine_i)
        open_times = OPEN_TIMES
        closed_times = CLOSED_TIMES
        
        if machine_i >=K2 and machine_i < K3:
            open_times = meta_infos["Open time second"]
            closed_times = meta_infos["Closed time second"]
        if machine_i >= K3:
            open_times = meta_infos["Open time thrid"]
            closed_times = meta_infos["Closed time thrid"]

        for i in range(len(update_route_info)):
            #print(update_route_info[i])
            start_day_work = ""
            start = str(startdate + timedelta(seconds=update_route_info[i]['start time']*60*60))
            arrive = str(startdate + timedelta(seconds=update_route_info[i]['arrive time']*60*60))
            if i == 0:
                last_end = start
                update_route_info[i]['arrive time'] = update_route_info[i]['start time']
            end_time = update_route_info[i]['end time']
            if type([]) == type(update_route_info[i]['end time']):
                end = str(startdate + timedelta(seconds=update_route_info[i]['end time'][-1]*60*60))
                end_time = update_route_info[i]['end time'][-1]
            else:
                end = str(startdate + timedelta(seconds=update_route_info[i]['end time']*60*60))

            
            if update_route_info[i]['arrive time'] != update_route_info[i]['start time']:
                delta_time = update_route_info[i]['arrive time'] - update_route_info[i]['start time']
                if  delta_time< 16:
                    df.append(dict(Task=machine, Subtask="Wait", Start=arrive, Finish=start) )
                    end_day_work = end
                    start_day_work = start
                else:
                    t_start = (update_route_info[i]['start time']//24)*24 + End_Times[k]
                    t_end= (update_route_info[i]['arrive time']//24)*24 + Start_Times[k]
                    end_day_work = str(startdate + timedelta(seconds= t_end*60*60))
                    start_day_work = str(startdate + timedelta(seconds=t_start*60*60))
                    df.append(dict(Task=machine, Subtask="Wait", Start=arrive, Finish=end_day_work) )
                    df.append(dict(Task=machine, Subtask="Wait-2", Start=end_day_work, Finish=start_day_work) )
                    df.append(dict(Task=machine, Subtask="Wait", Start=start_day_work, Finish=start) )

            delta_time = end_time - update_route_info[i]['start time']
            #print(delta_time)
            if  delta_time< 10:
                end_day_work = end
                start_day_work = start
                df.append(dict(Task=machine, Subtask="Operate-"+str(ty+1), Start=start, Finish=end) )
            else:
                t_end = (update_route_info[i]['start time']//24)*24 + End_Times[k]
                t_start = (end_time//24)*24 + Start_Times[k]
                end_day_work = str(startdate + timedelta(seconds= t_end*60*60))
                start_day_work = str(startdate + timedelta(seconds=t_start*60*60))
                df.append(dict(Task=machine, Subtask="Operate-"+str(ty+1), Start=start, Finish=end_day_work) )
                df.append(dict(Task=machine, Subtask="Wait-2", Start=end_day_work, Finish=start_day_work) )
                df.append(dict(Task=machine, Subtask="Operate-"+str(ty+1), Start=start_day_work, Finish=end) )
                #print("here", delta_time, End_Times[k])

            
            if i != 0:
                df.append(dict(Task=machine, Subtask="Travel", Start=last_end, Finish=arrive) )
            last_end = end
            name = update_route_info[i]['name'].replace("Feild0", "F")
            name = name.replace("Feild", "F")
            annots.append(dict(x=start,y=M-1-k + 0.3,text=name, showarrow=False, font=dict(color='Black')))
            machine_i= update_route_info[i]['machine']
            fid = update_route_info[i]['Farm ID']
            open_date = str(startdate + timedelta(seconds= open_times[fid]*60*60)).split()[0]
            closed_date = str(startdate + timedelta(seconds= (5*24+closed_times[fid])*60*60)).split()[0]
            str_start = ""
            
            if start_day_work != None:
                str_start = (str(start_day_work)).split()[0]
            row = ["S1",
                 Machine_IDs[machine_i],
                name,
                update_route_info[i]["LAT"],
                update_route_info[i]["LNG"],
                update_route_info[i]["machine type"][0],
                open_date,
                closed_date,
                str_start,
                update_route_info[i]["km"],

                str(timedelta(seconds= update_route_info[i]["travel time"]*60*60)),
                (str(start)).split()[1][:-3],
                str(timedelta(seconds= update_route_info[i]["operation time"]*60*60)),
                end[:-3],]
            
            temp = {
                "solution": "S1",
                "route_order": i,
                "machine_id": Machine_IDs[machine_i],
                "feild_id": IDs[update_route_info[i]["Farm ID"]],
                "feild_name": name,
                "lat": update_route_info[i]["LAT"],
                "lng": update_route_info[i]["LNG"],
                "machine_type": update_route_info[i]["machine type"][0],
                "open_date": open_date,
                "closed_date":closed_date,
                "work_date": str_start,
                "distance": update_route_info[i]["km"],

                "travel_time": str(timedelta(seconds= update_route_info[i]["travel time"]*60*60)),
                "start_time": (str(start)).split()[1][:-3],
                "operation_time":str(timedelta(seconds= update_route_info[i]["operation time"]*60*60)),
                "exit_time":end[:-3],
            }
            rows.append(row)
            jdata = temp.copy()
            json_datas.append(jdata)

    return update_route_infos, json_datas

if __name__ == "__main__":
    get_dataset("U3", "D1")