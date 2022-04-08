import re

import numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

np.set_printoptions(threshold=np.inf)
import pandas as pd

def caluculateDistance(chain,antigen, list):
    for x in range(len(collection[antigen])):
        for i in range(len(collection[chain])):
            a = np.array(([collection[chain][i]][0][0], [collection[chain][i]][0][1], [collection[chain][i]][0][2]))
            b = np.array(([collection[antigen][x]][0][0], [collection[antigen][x]][0][1], [collection[antigen][x]][0][2]))

            list[i][x]= np.linalg.norm(a - b).round(2)
    
    

f = open("3hfm.pdb", "r")

Lines = f.readlines()

counter = 0;

collection = []
list = []
collection.append(list)

with open("3hfm.pdb") as f:
    for line in f:
        if (line[0] + line[1] + line[2] == "TER"):
            saveList = list.copy()
            collection.append(saveList)
            counter = counter + 1
            list.clear()
        elif (line[0] + line[1] + line[2] + line[3] == "ATOM"):
            if (line.__contains__("CA")):
                formatedString = line[30:]
                clean = formatedString[:25]

                list.append(np.array([float(clean.split()[0]), float(clean.split()[1]), float(clean.split()[2])]))

#Light chain
print("Light chain = " ,collection[1])
print("Size ", len(collection[1]))
#Heavy chain
print("Heavy chain = " ,collection[2])
print("Size ", len(collection[2]))
#Antigen
print("Antigen = " , collection[3])
print("Size ", len(collection[3]))

lightChainDistanceToAntigen = numpy.zeros(shape=(len(collection[3]),len(collection[1])))


caluculateDistance(3,1,lightChainDistanceToAntigen)


print("Distance from light chain to Antigen = ", lightChainDistanceToAntigen)
print(len(lightChainDistanceToAntigen))

heavyChainDistanceToAntigen = numpy.zeros(shape=(len(collection[3]),len(collection[2])))
caluculateDistance(3,2,heavyChainDistanceToAntigen)


print("Distance from heavy chain to Antigen = ", heavyChainDistanceToAntigen)
print(len(heavyChainDistanceToAntigen))


fig = px.imshow(lightChainDistanceToAntigen)
fig.show()

fig2 = px.imshow(heavyChainDistanceToAntigen)
fig2.show()
