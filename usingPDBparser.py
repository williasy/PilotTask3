from Bio.PDB import *
import numpy
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

#np.set_printoptions(threshold=np.inf)

def caluculateDistance(chain,antigen, list):
    for x in range(len(collection[antigen])):
        for i in range(len(collection[chain])):
            a = np.array((collection[chain][i][0], collection[chain][i][1], collection[chain][i][2]))
            b = np.array((collection[antigen][x][0], collection[antigen][x][1], collection[antigen][x][2]))

            rounded = round(np.linalg.norm(a - b),2)
            list[x][i] = rounded


pdbParser = PDBParser(
    PERMISSIVE=True
)
structure = pdbParser.get_structure("pdbStructure", "3hfm.pdb")

print(pdbParser.get_header())

collection = []
list = []


counter = 0

for model in structure:
    for chain in model:
        residueCounter = 0
        for residue in chain:
            if residueCounter == 0:
                list.append(residue.get_parent())
                residueCounter = 1
            for atom in residue:
                if(atom.get_name() == "CA"):
                    list.append(atom.get_coord())

        saveList = list.copy()
        collection.append(saveList)
        counter = counter + 1
        list.clear()



#print("Light chain = " ,collection[0])
#print("Size ", len(collection[0])-1)
#Heavy chain
#print("Heavy chain = " ,collection[1])
#print("Size ", len(collection[1])-1)
#Antigen
#print("Antigen = " , collection[2])
#print("Size ", len(collection[2])-1)

for i in range(len(collection)):
    chainID = collection[i][0].copy()
    print(chainID, " = " , collection[i], "\nSize = ", len(collection[i])-1)


#Remove Chain ID from lists
for list in collection:
    list.pop(0)

lightChainDistanceToAntigen = numpy.zeros(shape=(len(collection[2]),len(collection[0])))

caluculateDistance(0,2,lightChainDistanceToAntigen)

print("Distance from light chain to Antigen = ", lightChainDistanceToAntigen)
print(len(lightChainDistanceToAntigen))

heavyChainDistanceToAntigen = numpy.zeros(shape=(len(collection[2]),len(collection[1])))
caluculateDistance(1,2,heavyChainDistanceToAntigen)


print("Distance from heavy chain to Antigen = ", heavyChainDistanceToAntigen)
print(len(heavyChainDistanceToAntigen))


fig = px.imshow(lightChainDistanceToAntigen)
fig.show()

fig2 = px.imshow(heavyChainDistanceToAntigen)
fig2.show()
