from numpy.core.memmap import uint8
import numpy as np
import time
import json
from datetime import datetime
from utility import get_dataset, create_json, get_sub_dataset
from problem import *
from test_utility import *
from decoder import *
from greedy_decoder import *
import pandas as pd

class AdaptiveWeight:
    def __init__(self, N, Max_iter, F, K):
        self.Ns = np.zeros(( Max_iter+1, N))
        self.As = np.zeros(( Max_iter+1, N))
        self.Is = np.zeros(( Max_iter+1, N))
        self.Iter = 0
        self.Max_iter = Max_iter
        self.N_track = N
        self.pobWeights = np.ones((N,))/N
        self.weights = np.ones((N,))
        self.Ns[0] = np.ones((N,))
        self.As[0] = np.ones((N,))
        self.Is[0] = np.ones((N,))
        self.F = F
        self.K = K
        self.listIndexs = list(np.arange(N, dtype=np.uint8))

    def random_tracks(self, size):
        tracksIds = np.random.choice(self.listIndexs, size, p=self.pobWeights)
        return tracksIds

    def update(self, tracks, objects, bestI):
        self.Iter += 1
        self.Is[self.Iter, bestI] = 1
        for i in range(self.N_track):
            indexs = np.where(tracks==i)[0]
            self.Ns[self.Iter, i] = len(indexs)
            self.As[self.Iter, i] = np.mean(objects[indexs])
            if len(indexs) == 0:
                self.As[self.Iter, i] = 0
            Ii = np.sum(self.Is[:self.Iter, i])

            new_weight = (self.F*self.Ns[self.Iter, i] + 
                              (1-self.F)*self.As[self.Iter, i] + self.K*Ii)
            
            #print("new_weight", new_weight)
            new_weight = max(new_weight, 0)
            self.weights[i] =  max(0.3*self.weights[i] + 0.7*new_weight, 0)
        total = np.sum(self.weights)
        #print(self.weights, total)
        for i in range(self.N_track):
            self.pobWeights[i] = self.weights[i]/total


class AMIS:
    def __init__(self, problem, pop_size,  max_iter, compute_time, CR=0.3, CRT=0.2, F1=0.6, F2=0.4, lo=0.5):
        self.pop_size = pop_size
        self.CR = CR
        self.problem = problem
        self.max_iter = max_iter
        self.Xs = None
        self.fitnessXs = None
        self.bestX = None
        self.best2X = None
        self.CRT = CRT
        self.bestFitness = 0
        self.F1 = F1
        self.F2 = F2
        self.lo = lo
        self.data = np.zeros((3*max_iter, self.problem.n_var))
        self.out = {}
        self.initialize()
        self.N_blackboxs = 6
        self.adaptiveWeight = AdaptiveWeight(self.N_blackboxs, max_iter, 0.7, 1)
        self.compute_time = compute_time

    def initialize(self):
        self.Xs = np.random.rand( self.pop_size, self.problem.n_var)
        self.fitnessXs = np.zeros((self.pop_size,))
        self.cal_fitness()
        max_index = np.argmax(self.fitnessXs)
        self.bestX = np.copy(self.Xs[max_index])
        self.best2X = np.copy(self.bestX)
        self.bestFitness = self.fitnessXs[max_index]
        
    def cal_fitness(self):
        for i in range(self.pop_size):
            self.problem._evaluate(self.Xs[i], self.out)
            self.fitnessXs[i] = self.out['F']
        
    def IB1_best(self, xs):
        N = len(xs)
        indexs = np.arange(0, self.pop_size, 1).astype(dtype=np.uint8)
        index_r1s = list(np.copy(indexs))
        index_r2s = list(np.copy(indexs))
        index_r3s = list(np.copy(indexs))
        np.random.shuffle(index_r1s)
        np.random.shuffle(index_r2s)
        np.random.shuffle(index_r3s)
        Vs = self.lo*self.Xs[index_r1s[:N]] + self.F1*(self.bestX - self.Xs[index_r1s[:N]]) +\
                                              self.F2*(self.Xs[index_r2s[:N]] - self.Xs[index_r3s[:N]])
        
        vmin = np.min(Vs)
        vmax = np.max(Vs)
        if vmin != vmax:
            Vs = (Vs-vmin)/(vmax-vmin)
        return Vs


    def IB2_best2(self, xs):
        N = len(xs)
        indexs = np.arange(0, self.pop_size, 1).astype(dtype=np.uint8)
        index_r1s = list(np.copy(indexs))
        index_r2s = list(np.copy(indexs))
        index_r3s = list(np.copy(indexs))
        np.random.shuffle(index_r1s)
        np.random.shuffle(index_r2s)
        np.random.shuffle(index_r3s)
        Vs = self.Xs[index_r1s[:N]] + self.F1*(self.bestX - self.Xs[index_r1s[:N]]) +\
                                              self.F2*(self.best2X - self.Xs[index_r1s[:N]])
        
        vmin = np.min(Vs)
        vmax = np.max(Vs)
        if vmin != vmax:
            Vs = (Vs-vmin)/(vmax-vmin)
        return Vs

    def IB3_DE(self, xs):
        N = len(xs)
        indexs = np.arange(0, self.pop_size, 1).astype(dtype=np.uint8)
        index_r1s = list(np.copy(indexs))
        index_r2s = list(np.copy(indexs))
        index_r3s = list(np.copy(indexs))
        np.random.shuffle(index_r1s)
        np.random.shuffle(index_r2s)
        np.random.shuffle(index_r3s)
        Vs = self.Xs[index_r1s[:N]] + self.F1*(self.best2X - self.Xs[index_r1s[:N]])
        
        vmin = np.min(Vs)
        vmax = np.max(Vs)
        if vmin != vmax:
            Vs = (Vs-vmin)/(vmax-vmin)
        return Vs

    def IB4_DER(self, xs):
        N = len(xs)
        indexs = np.arange(0, self.pop_size, 1).astype(dtype=np.uint8)
        index_r1s = list(np.copy(indexs))
        index_r2s = list(np.copy(indexs))
        index_r3s = list(np.copy(indexs))
        np.random.shuffle(index_r1s)
        np.random.shuffle(index_r2s)
        np.random.shuffle(index_r3s)
        Vs = self.Xs[index_r1s[:N]] + np.random.rand()*(self.best2X - self.Xs[index_r1s[:N]])
        
        vmin = np.min(Vs)
        vmax = np.max(Vs)
        if vmin != vmax:
            Vs = (Vs-vmin)/(vmax-vmin)
        return Vs

    def IB5_random(self, xs):
        N = len(xs)
        rs = np.random.rand(xs.shape[0], xs.shape[1])
        cross_select = np.random.choice(a=[0, 1], size=xs.shape, p=[self.CRT, 1-self.CRT])  
        Vs = rs
        return Vs
    
    
    def IB6_random_transit(self, xs):
        N = len(xs)
        rs = np.random.rand(xs.shape[0], xs.shape[1])
        cross_select = np.random.choice(a=[0, 1], size=xs.shape, p=[self.CRT, 1-self.CRT])  
        Vs = (1-cross_select)*xs + cross_select*rs
        return Vs
    
    def IB7_best_transit(self, xs):
        N = len(xs)
        rs = np.random.rand(xs.shape[0], xs.shape[1])
        cross_select = np.random.choice(a=[0, 1], size=xs.shape, p=[self.CRT, 1-self.CRT])  
        Vs = (1-cross_select)*xs + cross_select*self.bestX
        return Vs

    def IB8_scale(self, xs):
        N = len(xs)
        rs = np.random.rand(xs.shape[0], xs.shape[1])
        cross_select = np.random.choice(a=[0, 1], size=xs.shape, p=[self.CRT, 1-self.CRT])  
        Vs =  ((1 - cross_select)*rs)*xs +   cross_select*xs
        return Vs

    def IB9_select_another(self, xs):
        N = len(xs)
        indexs = np.arange(0, self.pop_size, 1).astype(dtype=np.uint8)
        index_r1s = list(np.copy(indexs))
        index_r2s = list(np.copy(indexs))
        np.random.shuffle(index_r1s)
        np.random.shuffle(index_r2s)
        cross_select = np.random.choice(a=[0, 1], size=xs.shape, p=[self.CRT, 1-self.CRT])  
        Vs =  ((1 - cross_select)*self.Xs[index_r1s[:N]])*xs +   cross_select*self.Xs[index_r2s[:N]]
        return Vs

    def select_blocks(self):
        self.Vs = np.copy(self.Xs)
        for i in range(self.N_blackboxs):
            indexI = list(np.where(self.currentTracks == i)[0])
            if len(indexI) == 0:
                continue
            if i == 0:
                self.Vs[indexI] = self.IB3_DE(self.Xs[indexI])
            elif i==1:
                self.Vs[indexI] = self.IB6_random_transit(self.Xs[indexI])
            elif i==2:
                self.Vs[indexI] = self.IB9_select_another(self.Xs[indexI])
            elif i==3:
                self.Vs[indexI] = self.IB4_DER(self.Xs[indexI])
            elif i==4:
                self.Vs[indexI] = self.IB2_best2(self.Xs[indexI])
            elif i==5:
                self.Vs[indexI] = self.IB1_best(self.Xs[indexI])

    def recombination(self):
        cross_select = np.random.choice(a=[0, 1], size=self.Xs.shape, p=[self.CR, 1-self.CR])  
        self.Us = (1-cross_select)*self.Xs + cross_select*self.Vs
        
    def accept_all(self):
        fitnessUs = np.zeros((self.pop_size,))
        select_mask = np.zeros((self.pop_size,))
        for i in range(self.pop_size):
            self.problem._evaluate(self.Us[i], self.out)
            fitnessUs[i] = self.out['F']
            self.fitnessXs[i] = fitnessUs[i]
            self.Xs[i] = np.copy(self.Us[i])
            if fitnessUs[i] > self.bestFitness:
                self.bestFitness = fitnessUs[i]
                self.best2X = np.copy(self.bestX)
                self.bestX = np.copy(self.Us[i])

    def selection(self):
        fitnessUs = np.zeros((self.pop_size,))
        select_mask = np.zeros((self.pop_size,))
        for i in range(self.pop_size):
            self.problem._evaluate(self.Us[i], self.out)
            fitnessUs[i] = self.out['F']
            delta = fitnessUs[i] - self.fitnessXs[i]
            if (delta < 0 or 
                np.random.rand() + 5 < np.exp(-delta*100) or
                np.random.rand()  < 0.00001):
                self.fitnessXs[i] = fitnessUs[i]
                self.Xs[i] = np.copy(self.Us[i])
                if fitnessUs[i] < self.bestFitness:
                    self.bestFitness = fitnessUs[i]
                    self.best2X = np.copy(self.bestX)
                    self.bestX = np.copy(self.Us[i])

    def single_iterate(self, it):
        self.currentTracks = self.adaptiveWeight.random_tracks(self.pop_size)
        self.select_blocks()
        arg_max = np.argmax(self.fitnessXs)
        self.adaptiveWeight.update(self.currentTracks, self.fitnessXs, self.currentTracks[arg_max])
        self.recombination()
        self.selection()

        idx= np.argsort(self.bestX)

        print(it, round(np.mean(self.fitnessXs), 4), round(self.bestFitness, 4), self.adaptiveWeight.pobWeights)
        arg_maxs = np.argsort(self.fitnessXs)
        self.data[it*3:3*(it+1)] = np.copy(self.Xs[arg_maxs[:3]])

    def iterate(self):
        print(self.adaptiveWeight.pobWeights)
        start_time = time.time()
        for it in range(self.max_iter):
            self.single_iterate(it)
            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed > self.compute_time:
                print("End by Time")
                break
            else:
                print(f"time: {elapsed}")
            

def test():
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
    
    f = open("/root/sugar_route/data_test/data_U53.json")
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

        route_decoder = Decoder(dataset_lookups)
        print(dataset_lookups['Feild_ID'])
        print(dataset_lookups['Machine_ID'])

        #for feild in field_ids:
            #print(feild)


        sugar_problem = SurgarProblem(dataset_lookups)
        np.random.seed(0)
        compute_time = 100
        algorithm = AMIS(sugar_problem,
            pop_size=100, # 1 จำนวนคำตอบต่อรอบ
            CR=0.3,
            max_iter = 3, # 2 จำนวนรอบ
            #max_iter  = 2,
            compute_time = compute_time,
            #max_iter = 10,
            #dither="vector",
            #jitter=False
        )
        meta_infos = route_decoder.decode(algorithm.bestX, True)
        algorithm.iterate()

        update_route_infos, json_datas, cost_datas, df, annots = create_json(sugar_problem.dataset_lookups, 
                                                                            route_decoder, algorithm.bestX)
        print("------------------------------------------------------------------------")
        meta_infos = route_decoder.decode(algorithm.bestX, True)
        rinfos1, rinfos2, rinfos3, rinfos4  = meta_infos['route_sweeper'], meta_infos['route_baler'], meta_infos['route_picker'], meta_infos[ 'route_truck']


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
                print(dataset_lookups['Feild_ID'][rinfo['route']], rinfo['ds'])
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
        zone_results[0]
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


def test_create_solution():
    name = 'data_U53'
    dataset_lookups = creaet_data_lookup_test(f'data_test/{name}.json')
    decoder = GreedyDecoder(dataset_lookups)
    sugar_problem = SurgarProblem(dataset_lookups)
    compute_time = 30
    np.random.seed(0)
    algorithm = AMIS(sugar_problem,
            pop_size=3, # 1 จำนวนคำตอบต่อรอบ
            CR=0.3,
            max_iter = 1, # 2 จำนวนรอบ
            #max_iter  = 2,
            compute_time = compute_time,
            #max_iter = 10,
            #dither="vector",
            #jitter=False
        )
    
    algorithm.iterate()
    print("-------------ENd------------------")
    meta_infos = decoder.decode(algorithm.bestX, True)
    rinfos1, rinfos2, rinfos3, rinfos4  = meta_infos['route_sweeper'], meta_infos['route_baler'], meta_infos['route_picker'], meta_infos[ 'route_truck']
    
    update_route_infos, json_datas, cost_datas, df, annots = create_json(sugar_problem.dataset_lookups, 
                                                                            decoder, algorithm.bestX)

    route_infos = [rinfos1, rinfos2, rinfos3, rinfos4]
    #print(dataset_lookups['Feild_ID'])
    for rinfos in route_infos:
        for rinfo in rinfos:
            #pass
            #print(rinfo)
            #print(dataset_lookups['Feild_ID'][rinfo['route']])
            print(rinfo['machine'], dataset_lookups['Feild_ID'][rinfo['route']], rinfo['ds'])
            pass
    
    df = pd.DataFrame(json_datas)
    df.to_csv(f"/root/sugar_route/{name}.csv", index=False)
    df = pd.DataFrame(cost_datas['machine_costs'])
    df.to_csv(f"/root/sugar_route/cost_table_{name}.csv", index=False)

def open_closed_times():
    name = 'data_single'
    dataset_lookups = creaet_data_lookup_test(f'{name}.json')
    decoder = GreedyDecoder(dataset_lookups)
    
    N = dataset_lookups["Number_Feilds"]
    NF = dataset_lookups["Number_Feilds_NF"]
    Open_days = dataset_lookups["OPEN_DAY"]



if __name__ == "__main__":
    pass
    test_create_solution()
  