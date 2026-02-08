from input import getInput

graph={}
vertices,edges=0,0
graph, vertices, edges = getInput()
    

def bellman_ford(graph, source):
    distances=[]
    for vertex in range(1, vertices + 1):
        distances.append(float('infinity'))
    distances[source] = 0  

    # Relax edges repeatedly
    for _ in range(vertices - 1):
        for u in graph:           # u=current vertex
            for v, w in graph[u]: # v=neighbor, w=weight

                # Relaxation step
                if distances[u - 1] + w < distances[v - 1]:
                    distances[v - 1] = distances[u - 1] + w


    # Check for negative-weight cycles
    for u in graph: 
        for v, w in graph[u]:
            if distances[u - 1] + w < distances[v - 1]:
                print("Graph contains negative weight cycle")
                return None

    return distances


print("\n RUNNING BELLMAN-FORD ALGORITHM ...")
start_vertex = 1
distances = bellman_ford(graph, start_vertex - 1)
if distances:
    print("Shortest distances from vertex", start_vertex, "using Bellman-Ford:")
    for vertex in range(1, vertices + 1):
        print("Vertex", vertex, ":", distances[vertex - 1])
