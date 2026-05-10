import heapq
from collections import defaultdict

def dijkstra(graph, start):
    """
    Find the shortest paths from a starting node to all other nodes using Dijkstra's algorithm.

    Args:
        graph: A dictionary where keys are node identifiers and values are lists
               of (neighbor_node, weight) tuples.
        start: The starting node from which shortest paths are calculated.

    Returns:
        A dictionary mapping each node to its shortest distance from the start node.
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances


def dijkstra_with_path(graph, start):
    """
    Find shortest paths and allow path reconstruction.

    Args:
        graph: A dictionary where keys are node identifiers and values are lists
               of (neighbor_node, weight) tuples.
        start: The starting node from which shortest paths are calculated.

    Returns:
        A tuple containing:
            - distances: dictionary of shortest distances from start node.
            - previous: dictionary of previous node for each node (used to reconstruct paths).
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    pq = [(0, start)]
    visited = set()

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous


def reconstruct_path(previous, start, end):
    """
    Reconstruct the shortest path between two nodes using the 'previous' dictionary.

    Args:
        previous: Dictionary from dijkstra_with_path()
        start: Starting node (used for verification)
        end: Ending node for path reconstruction

    Returns:
        List of nodes representing the shortest path from start to end.
    """
    path = []
    node = end
    while node is not None:
        path.append(node)

        node = previous.get(node)

    return path[::-1]


def draw_graph(graph):
    """
    Simple function to print graph structure for readability.
    """
    print("\nGraph Structure (Adjacency List):")
    for node, edges in graph.items():
        edges_str = ', '.join(f"{neighbor} (weight: {weight})" for neighbor, weight in edges)
        print(f"{node}: {edges_str}")


# Example Usage
if __name__ == "__main__":
    # Create a sample graph (adjacency list format)
    graph = {
        'A': [('B', 1), ('C', 4), ('D', 5)],
        'B': [('D', 2), ('E', 6)],
        'C': [('B', 3), ('D', 8), ('E', 9)],
        'D': [('C', 6), ('E', 3)],
        'E': []
    }

    draw_graph(graph)

    start_node = 'A'
    print(f"\nCalculating shortest paths from start node: {start_node}")
    print("=" * 50)

    # Calculate shortest distances
    shortest_distances = dijkstra(graph, start_node)

    # Calculate shortest paths with path tracking
    distances, previous = dijkstra_with_path(graph, start_node)

    # Display results
    print("\nShortest Distances from node", start_node, ":")
    print("-" * 30)
    for node, distance in sorted(distances.items()):
        print(f"{node:6s}: {distance:6.1f}")

    # Display paths for some nodes
    print("\nShortest Paths from node", start_node, ":")
    print("-" * 30)
    targets = ['B', 'C', 'D', 'E']

    for target in targets:
        path = reconstruct_path(previous, start_node, target)
        path_str = " -> ".join(path)
        print(f"Path to {target:2s}: {path_str:25s} | Distance: {distances[target]:6.1f}")

    # Optional: Check reachability
    unreachable = [node for node, dist in distances.items() if dist == float('inf')]
    if unreachable:
        print("\nUnreachable nodes from start:", unreachable)
    else:
        print("\nAll nodes are reachable from the start node.")

    print("\n" + "=" * 50)
    print("Program completed successfully!")
