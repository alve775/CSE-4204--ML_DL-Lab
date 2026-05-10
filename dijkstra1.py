import heapq


def dijkstra(graph, start_node):
    """
    Implements Dijkstra's algorithm to find the shortest paths from a start_node
    to all other nodes in a weighted graph.

    Args:
        graph (dict): The graph represented as an adjacency list.
                       Format: {node: {neighbor: weight, ...}, ...}
        start_node: The node from which to start the search.

    Returns:
        tuple: (distances, predecessors) where:
               - distances: A dictionary of the shortest distance from start_node to every other node.
               - predecessors: A dictionary used to reconstruct the path.
    """
    # 1. Initialization
    # Distances dictionary stores the shortest known distance from the start_node to every other node.
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0

    # Predecessors dictionary helps reconstruct the path later.
    predecessors = {node: None for node in graph}

    # Priority Queue (Min-Heap): Stores tuples of (distance, node).
    # We start by pushing the start_node with distance 0.
    priority_queue = [(0, start_node)]

    # Set to keep track of nodes whose shortest path has been finalized (optional but good practice)
    # visited = set()

    # 2. Main Loop
    while priority_queue:
        # Get the node with the smallest current distance from the queue
        current_distance, current_node = heapq.heappop(priority_queue)

        # Optimization: If we pull an outdated entry from the heap, skip it.
        # This happens because we might push a node multiple times with different distances.
        if current_distance > distances[current_node]:
            continue

        # Explore neighbors
        for neighbor, weight in graph[current_node].items():
            # Calculate the distance to the neighbor through the current node
            distance = current_distance + weight

            # Relaxation Step: If a shorter path is found, update the distance and push to the queue
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                # Push the newly found shorter path onto the heap
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors


def reconstruct_path(predecessors, start_node, end_node):
    """Reconstructs the shortest path from the predecessors map."""
    path = []
    current = end_node
    while current is not None:
        path.insert(0, current)
        if current == start_node:
            break
        current = predecessors[current]
    return path if path and path[0] == start_node else []


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Define the graph using an adjacency list structure.
    # The structure is: {Node: {Neighbor: Weight}}
    graph_data = {
        'A': {'B': 2, 'C': 3},
        'B': {'A': 2, 'D': 4, 'E': 2},
        'C': {'A': 3, 'D': 1, 'E': 4},
        'D': {'C': 1, 'F': 5},
        'E': {'B': 2, 'C': 4, 'F': 3},
        'F': {'D': 5, 'E': 3}
    }

    start = 'A'
    print(f"--- Running Dijkstra's Algorithm starting from Node '{start}' ---")

    # 1. Run the algorithm
    shortest_distances, predecessors = dijkstra(graph_data, start)

    # 2. Display Results
    print("\n=========================================")
    print("Shortest Distances from Start Node:")
    for node, dist in shortest_distances.items():
        print(f"  To {node}: {dist}")
    print("=========================================")

    # 3. Example Path Reconstruction
    end_node = 'F'
    print(f"\n--- Finding Shortest Path from {start} to {end_node} ---")

    path = reconstruct_path(predecessors, start, end_node)

    if path:
        print(f"Shortest Path found: {' -> '.join(path)}")
        # The distance to F is stored in the distances map
        print(f"Total minimum cost: {shortest_distances[end_node]}")
    else:
        print(f"No path exists from {start} to {end_node}.")