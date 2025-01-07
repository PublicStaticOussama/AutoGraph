from collections import deque

def DFS(node):
    if not len(node.vars):
        return [node]
    ordered_vars = []
    for root in node.vars:
        ordered_vars += DFS(root)
    ordered_vars.append(node)
    return ordered_vars

def topological_DFS(node, visited=None, ordered_nodes=None, levels=None, level=None, key=lambda x: x.vars):
    if visited is None:
        visited = set()
    if ordered_nodes is None:
        ordered_nodes = []
    if levels is None:
        levels = []
    if level is None:
        level = 0

    if node not in visited:
        for root in key(node):
            topological_DFS(root, visited, ordered_nodes, levels, level + 1, key=key)
        ordered_nodes.append(node)
        levels.append(level)
        visited.add(node)
    return ordered_nodes, levels

def branchless_DFS(node, visited=None, ordered_nodes=None, levels=None, level=None, key=lambda x: x.vars):
    if visited is None:
        visited = set()
    if ordered_nodes is None:
        ordered_nodes = []
    if levels is None:
        levels = []
    if level is None:
        level = 0

    if node not in visited:
        key_roots = key(node)
        if node.op == "merge":
            key_roots = [key_roots[not node.kwarge["condition"]]]
        for root in key_roots:
            topological_DFS(root, visited, ordered_nodes, levels, level + 1, key=key)
        ordered_nodes.append(node)
        levels.append(level)
        visited.add(node)
    return ordered_nodes, levels

def BFS(node):
    """
    Perform BFS traversal on a graph represented by OpNode.

    Parameters:
    - root: The starting node (OpNode).

    Returns:
    - result: A list of node names in BFS order.
    """
    if node is None:
        return []
    
    visited = set()
    queue = deque([node])
    result = []

    while queue:
        current_node = queue.popleft()
        if current_node not in visited:
            # Process current node
            result.append(current_node)
            visited.add(current_node)

            # Add children to the queue
            for child in current_node.vars:
                if child not in visited:
                    queue.append(child)

    return result

