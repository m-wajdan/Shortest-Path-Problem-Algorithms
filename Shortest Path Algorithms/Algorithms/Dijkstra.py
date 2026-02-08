from input import getInput

graph={}
vertices,edges=0,0
graph, vertices, edges = getInput()

def dijkstra(graph, start):
    heap =[] 
    heap.append((0, start))  # (distance, vertex)
    distances = {}
    for vertex in range(1, vertices + 1):
        distances[vertex] = float('infinity')
    distances[start] = 0

    while heap:
        current_distance, current_vertex = heap.pop(0)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph.get(current_vertex, []):
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heap.append((distance, neighbor))
                heap.sort()  # Maintain the heap property (Min-Heap)
    return distances

print("\n RUNNING DIJKSTRA'S ALGORITHM ...")
start_vertex = 1
distances = dijkstra(graph, start_vertex)
print(f"Shortest distances from vertex {start_vertex}:")
for vertex in range(1, vertices + 1):
    print(f"Vertex {vertex}: {distances[vertex]}")
