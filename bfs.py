from collections import deque


def bfs(graph, start):
    """
    Perform Breadth-First Search on a graph.

    Args:
        graph: Dictionary adjacency list in the form:
               {node: [neighbor1, neighbor2, ...]}
        start: The node where BFS starts.

    Returns:
        A list containing nodes in BFS traversal order.
    """
    if start not in graph:
        return []

    visited = set()
    queue = deque([start])
    traversal_order = []

    while queue:
        current_node = queue.popleft()

        if current_node in visited:
            continue

        visited.add(current_node)
        traversal_order.append(current_node)

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                queue.append(neighbor)

    return traversal_order


def bfs_shortest_path(graph, start, target):
    """
    Find the shortest path between two nodes in an unweighted graph using BFS.

    Returns:
        A list of nodes representing the shortest path.
        Returns an empty list if no path exists.
    """
    if start not in graph or target not in graph:
        return []

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        current_node, path = queue.popleft()

        if current_node == target:
            return path

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return []


if __name__ == "__main__":
    sample_graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B"],
        "E": ["B", "F"],
        "F": ["C", "E"],
    }

    start_node = "A"
    target_node = "F"

    print("BFS traversal:")
    print(bfs(sample_graph, start_node))

    print(f"\nShortest path from {start_node} to {target_node}:")
    print(bfs_shortest_path(sample_graph, start_node, target_node))
