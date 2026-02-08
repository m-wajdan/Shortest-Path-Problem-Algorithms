import pandas as pd
import numpy as np
import heapq
from helperFunctions import bellman_ford, dijkstra
from input import getInput

graph={}
vertices,edges=0,0
graph, vertices, edges = getInput()


def johnson(graph):
    #step 1: add new vertex '0' 
    newGraph={}
    newGraph[0] = []

    #step 2:  connected to all other vertices with edge weight 0
    for vertex in range(1, vertices + 1):
        newGraph[0].append((vertex, 0))
        newGraph[vertex] = graph.get(vertex, [])
    # step 3: run bellman ford from new vertex '0'
    bellmanDist= bellman_ford(newGraph, 0,vertices)
    if bellmanDist is None:
        print("Graph contains negative weight cycle. Johnson's algorithm cannot proceed.")
        return None
    # step 4: reweight edges
    reweightedGraph={}
    for u in graph:
        reweightedGraph[u] = []
        for v, w in graph[u]:
            newWeight = w + bellmanDist[u] - bellmanDist[v]
            reweightedGraph[u].append((v, newWeight))
    # step 5: remove the added vertex '0' (not needed anymore as we have reweighted the graph) 
    # step 6: run dijkstra for each vertex
    ActualDIst={}
    for v in range(1, vertices + 1):
        dijkstraDist = dijkstra(reweightedGraph, v,vertices)
        ActualDIst[v] = {}
        for u in dijkstraDist:
            ActualDIst[v][u] = dijkstraDist[u] - bellmanDist[v] + bellmanDist[u]
    return ActualDIst


def print_matrix_pandas(dist):
    # Convert INF to a readable "INF"
    df = pd.DataFrame(dist)
    df.replace([float('inf'), np.inf], "INF", inplace=True)

    # Rename rows and columns (1-based)
    df.index = [f"{i}" for i in range(1, len(dist) + 1)]
    df.columns = [f"{j}" for j in range(1, len(dist) + 1)]
    print("\nDistance Matrix:")
    print(df)


all_pairs_distances = johnson(graph)
if all_pairs_distances:
    print("\n RUNNING JOHNSON'S ALGORITHM ...")
    for u in all_pairs_distances:
        for v in all_pairs_distances[u]:
            print(f"Distance from vertex {u} to vertex {v}: {all_pairs_distances[u][v]}")

    # Convert to matrix form for better visualization
    distance_matrix = []
    for i in range(1, vertices + 1):
        row = []
        for j in range(1, vertices + 1):
            row.append(all_pairs_distances[i][j])
        distance_matrix.append(row)
    print_matrix_pandas(distance_matrix)
    