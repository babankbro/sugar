
from numpy.core.memmap import uint8
import numpy as np
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.de import DE
import numpy as np
from pymoo.core.problem import ElementwiseProblem
import math
from decoder import Decoder

class SurgarProblem(ElementwiseProblem):
    
    def __init__(self, dataset_lookups):
        self.dataset_lookups = dataset_lookups
        self.decoder = Decoder(dataset_lookups)
        NF = self.dataset_lookups["Number_Feilds_NF"]
        N = self.dataset_lookups["Number_Feilds"]
        self.N = 4*(N+NF)+4
        super().__init__(n_var=self.N, n_obj=1, n_constr=0, xl=0, xu=1)
        self.Prices = self.dataset_lookups["Price"]
        self.Areas = self.dataset_lookups["Area"]
        self.Fuel_Rates = self.dataset_lookups["Fuel_Rate"]
        self.Production_Rates = self.dataset_lookups["Production_Rate"]


    def _evaluate(self, x, out, *args, **kwargs):
        
        rinfos1, rinfos2, rinfos3, rinfos4,  _ = self.decoder.decode(x)
        if rinfos1 == None:
            cost = 1000000
        else:
        
            cost = 0
            profit = 0
            for rinfo in rinfos1:
                #print(rinfo)
                cost += np.sum(rinfo['ds'])*self.Fuel_Rates[rinfo['machine']]
                
            for rinfo in rinfos2:
                cost += np.sum(rinfo['ds'])*self.Fuel_Rates[rinfo['machine']]
            for rinfo in rinfos3:
                cost += np.sum(rinfo['ds'])*self.Fuel_Rates[rinfo['machine']]
            for rinfo in rinfos4:
                cost += np.sum(rinfo['ds'])*self.Fuel_Rates[rinfo['machine']]
                for f in rinfo['route']:
                    profit += self.Areas[f]*self.Production_Rates[f]*self.Prices[f]

        out["hash"] = hash(str(x))
        out["F"] = cost*30 - profit
        out["pheno"] = {"rating": 0, 'route':0, 'distance':cost, 'time':0}