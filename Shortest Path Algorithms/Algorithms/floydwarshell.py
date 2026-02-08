import pandas as pd
import numpy as np
from input import getInput

graph={}
vertices,edges=0,0
graph, vertices, edges = getInput()



def floydwarshell(graph):
    # step 1: Initialize distance matrix
    dist =[]
    for i in range(vertices):
        dist.append([])
        for j in range(vertices):
            if i == j:
                dist[i].append(0)
            else:
                dist[i].append(float('infinity'))

    #step 2: Initialize distances based on graph edges            
    for u in graph:
        for v, w in graph[u]:
            dist[u - 1][v - 1] = w

    # step 3: Update distance matrix
    for i in range(vertices):          # intermediate 
        for j in range(vertices):      # start       
            for k in range(vertices):  # end         

                # Relaxation step 
                if dist[j][i] + dist[i][k] < dist[j][k]:
                    dist[j][k] = dist[j][i] + dist[i][k]
    return dist

distances = floydwarshell(graph)

def print_matrix_pandas(dist):
    # Convert INF to a readable "INF"
    df = pd.DataFrame(dist)
    df.replace([float('inf'), np.inf], "INF", inplace=True)

    # Rename rows and columns (1-based)
    df.index = [f"{i}" for i in range(1, len(dist) + 1)]
    df.columns = [f"{j}" for j in range(1, len(dist) + 1)]

    print(df)

print("\n RUNNING FLOYD-WARSHALL ALGORITHM ...")
print_matrix_pandas(distances)
