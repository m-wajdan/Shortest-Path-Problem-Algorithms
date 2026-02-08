def bellman_ford(graph, source, vertices):
    distances = {}
    # Initialize all vertices that appear in the graph
    for u in graph:
        distances[u] = float('infinity')
        for v, w in graph[u]:
            if v not in distances:
                distances[v] = float('infinity')
    
    distances[source] = 0  

    # Relax edges repeatedly
    for _ in range(vertices - 1):
        for u in graph:
            if distances[u] != float('infinity'):
                for v, w in graph[u]:
                    # Relaxation step
                    if distances[u] + w < distances[v]:
                        distances[v] = distances[u] + w

    # Check for negative-weight cycles
    for u in graph: 
        if distances[u] != float('infinity'):
            for v, w in graph[u]:
                if distances[u] + w < distances[v]:
                    print("Graph contains negative weight cycle")
                    return None

    return distances

def dijkstra(graph, start,vertices):
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
