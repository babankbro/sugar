import numpy as np
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import utm
import math
import datetime
from variable import *



print(LATs)
print(LNGs)
plt.figure(figsize=(10, 10))
plt.scatter(LATs[:], LNGs[:], color='r', s=100)
plt.savefig('points.png')



from decoding import *
from AMIS import *


problem = SurgarProblem(4*(N+NF)+4)
np.random.seed(0)

algorithm = AMIS(problem,
    pop_size=100,
    CR=0.3,
    max_iter = 100,
    #dither="vector",
    #jitter=False
)
algorithm.iterate()

x = algorithm.bestX
rinfos1, rinfos2, rinfos3,rinfos4, meta_infos =decode(x)
meta_infos['Open time second']

np.save("result.npy", x)


print(x)
print()
for rinfo in rinfos1:
    print(rinfo["machine"], rinfo["route"])


    return update_route_infos, json_datasp