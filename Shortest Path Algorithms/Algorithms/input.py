def getInput():
    graph={}
    vertices,edges=0,0
    print("""Enter vertices, Edges and egde weight in the folloing format:
    Example Input Format:
    8     -> Number of lines to follow (edges + 1)
    5 7   -> Number of vertices and edges
    1 2 2 -> Edge from vertex 1 to vertex 2 with weight 2
    1 3 4
    2 3 1
    2 4 7
    3 5 3
    4 5 1
    5 4 2

    You are supposed to enter edges one by one in the same format as above.
    """)


    lines = iter(input().strip() for _ in range(int(input("Enter number of lines to follow (edges + 1): ")) ))

    first_line = next(lines)

    vertices, edges = map(int, first_line.split())

    for _ in range(edges):
        u, v, w = map(int, next(lines).split())
        if u not in graph:
            graph[u] = []
        graph[u].append((v, w))

    print("no of vertices:", vertices)
    print("no of edges:", edges)
    print("Graph representation (Adjacency List):")

    for vertex in graph:
        print("vertex", vertex, "->", graph[vertex])
    
    return graph, vertices, edges