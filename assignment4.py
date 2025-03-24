"""
Ayman Momin
UCID: 30192494
Assignment 4
CPSC 441
"""

# Import required libraries
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq

# Define all magical paths from the table (edges)
edges = [
    ('British Columbia', 'Saskatchewan', 1800, 19, 6),
    ('Alberta', 'Quebec', 2000, 21, 7),
    ('Ontario', 'Nova Scotia', 1300, 13, 4),
    ('Quebec', 'Newfoundland and Labrador', 1900, 20, 26),
    ('Nova Scotia', 'Saskatchewan', 1800, 18, 5),
    ('Alberta', 'Saskatchewan', 1600, 8, 3),
    ('Newfoundland and Labrador', 'Alberta', 2400, 24, 9),
    ('Ontario', 'Quebec', 500, 5, 1),
    ('Nova Scotia', 'Ontario', 2000, 21, 7),
    ('Saskatchewan', 'Nova Scotia', 2000, 20, 37),
    ('Quebec', 'Saskatchewan', 200, 2, 0),
    ('Alberta', 'Ottawa', 2400, 24, 9),
    ('Saskatchewan', 'Quebec', 2000, 20, 6),
    ('Ontario', 'Alberta', 1500, 16, 4),
    ('British Columbia', 'Saskatchewan', 1200, 14, 3),
    ('Newfoundland and Labrador', 'Quebec', 2200, 22, 7),
    ('Nova Scotia', 'Newfoundland and Labrador', 1200, 12, 6),
    ('Quebec', 'Ottawa', 1800, 19, 17),
    ('Alberta', 'British Columbia', 1800, 18, 27),
    ('British Columbia', 'Quebec', 1900, 19, 7),
    ('Ontario', 'Newfoundland and Labrador', 2300, 23, 8),
    ('Nova Scotia', 'Alberta', 2200, 22, 8),
    ('Newfoundland and Labrador', 'Alberta', 2300, 23, 8),
    ('Alberta', 'Newfoundland and Labrador', 2400, 24, 9),
    ('Saskatchewan', 'British Columbia', 2000, 21, 8),
    ('Ontario', 'Saskatchewan', 1600, 16, 5),
    ('Quebec', 'Nova Scotia', 1000, 10, 2),
    ('Newfoundland and Labrador', 'Saskatchewan', 2200, 23, 19),
    ('Nova Scotia', 'Quebec', 1100, 11, 2),
    ('British Columbia', 'Newfoundland and Labrador', 2500, 26, 10),
    ('Ontario', 'Ottawa', 1450, 4, 12),
    ('Alberta', 'Saskatchewan', 600, 8, 3),
    ('Quebec', 'Alberta', 1700, 17, 6),
    ('Saskatchewan', 'Nova Scotia', 1800, 18, 5),
    ('Alberta', 'Quebec', 2000, 21, 6),
    ('Nova Scotia', 'British Columbia', 2500, 26, 10),
    ('Ontario', 'Nova Scotia', 1300, 13, 4),
    ('British Columbia', 'Saskatchewan', 1800, 19, 6),
]

# Build the graph as an adjacency list
graph = {}
for edge in edges:
    start, end, distance, time, dementors = edge
    if start not in graph:
        graph[start] = []
    graph[start].append((end, distance, time, dementors))
    if end not in graph:
        graph[end] = []

# BFS for Shortest Hop Path (SHP)
def bfs_shortest_path(graph, start, end):
    visited = {start: None}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        if current == end:
            break
        for neighbor in graph[current]:
            next_node = neighbor[0]
            if next_node not in visited:
                visited[next_node] = current
                queue.append(next_node)
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = visited.get(current)
    path.reverse()
    return path if path and path[0] == start else []

# Dijkstra's Algorithm for SDP/STP/FDP
def dijkstra(graph, start, end, weight_idx):
    distances = {node: float('inf') for node in graph}
    prev_nodes = {node: None for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    
    while heap:
        curr_dist, curr_node = heapq.heappop(heap)
        if curr_node == end:
            break
        if curr_dist > distances[curr_node]:
            continue
        for neighbor in graph[curr_node]:
            next_node = neighbor[0]
            weight = neighbor[weight_idx]
            new_dist = curr_dist + weight
            if new_dist < distances[next_node]:
                distances[next_node] = new_dist
                prev_nodes[next_node] = curr_node
                heapq.heappush(heap, (new_dist, next_node))
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev_nodes.get(current)
    path.reverse()
    return path if path and path[0] == start else []

# Calculate total metric for a path
def calculate_total(graph, path, attr_idx):
    total = 0
    for i in range(len(path)-1):
        current = path[i]
        next_node = path[i+1]
        for neighbor in graph[current]:
            if neighbor[0] == next_node:
                total += neighbor[attr_idx]
                break
    return total

# Alumni data
alumni = {
    'Harry Potter': 'British Columbia',
    'Hermione Granger': 'Ontario',
    'Ron Weasley': 'Quebec',
    'Luna Lovegood': 'Newfoundland and Labrador',
    'Neville Longbottom': 'Saskatchewan',
    'Ginny Weasley': 'Nova Scotia'
}

# === Extra Credit: Combined Optimal Path ===
# Calculate max values for normalization
max_distance = max(edge[2] for edge in edges)
max_time = max(edge[3] for edge in edges)
max_dementors = max(edge[4] for edge in edges)
max_hops = 7  # Approximated maximum hops

# Weights for each criterion (equal)
weights = {
    'distance': 0.25,
    'time': 0.25,
    'dementors': 0.25,
    'hops': 0.25
}

# Build combined graph
graph_combined = {}
for edge in edges:
    start, end, distance, time, dementors = edge
    norm_distance = (distance / max_distance) * weights['distance']
    norm_time = (time / max_time) * weights['time']
    norm_dementors = (dementors / max_dementors) * weights['dementors']
    norm_hops = (1 / max_hops) * weights['hops']
    combined_weight = norm_distance + norm_time + norm_dementors + norm_hops
    
    if start not in graph_combined:
        graph_combined[start] = []
    graph_combined[start].append((end, combined_weight))
    if end not in graph_combined:
        graph_combined[end] = []

# Dijkstra's for combined path
def dijkstra_combined(graph, start, end):
    distances = {node: float('inf') for node in graph}
    prev_nodes = {node: None for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    
    while heap:
        curr_dist, curr_node = heapq.heappop(heap)
        if curr_node == end:
            break
        if curr_dist > distances[curr_node]:
            continue
        for neighbor, weight in graph[curr_node]:
            new_dist = curr_dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                prev_nodes[neighbor] = curr_node
                heapq.heappush(heap, (new_dist, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev_nodes.get(current)
    path.reverse()
    return path if path and path[0] == start else []

# Calculate combined score for a path
def calculate_combined_score(graph, path):
    total = 0
    for i in range(len(path)-1):
        current = path[i]
        next_node = path[i+1]
        for neighbor, weight in graph[current]:
            if neighbor == next_node:
                total += weight
                break
    return total

# Generate output for all alumni
for name, province in alumni.items():
    print(f"\n=== {name} ({province}) ===")
    
    # Shortest Hop Path (SHP)
    shp = bfs_shortest_path(graph, province, 'Ottawa')
    print(f"SHP: {' -> '.join(shp)} (Hops: {len(shp)-1})")
    
    # Shortest Distance Path (SDP)
    sdp = dijkstra(graph, province, 'Ottawa', 1)
    sdp_dist = calculate_total(graph, sdp, 1) if sdp else 0
    print(f"SDP: {' -> '.join(sdp)} (Distance: {sdp_dist} km)")
    
    # Shortest Time Path (STP)
    stp = dijkstra(graph, province, 'Ottawa', 2)
    stp_time = calculate_total(graph, stp, 2) if stp else 0
    print(f"STP: {' -> '.join(stp)} (Time: {stp_time} hrs)")
    
    # Fewest Dementors Path (FDP)
    fdp = dijkstra(graph, province, 'Ottawa', 3)
    fdp_dem = calculate_total(graph, fdp, 3) if fdp else 0
    print(f"FDP: {' -> '.join(fdp)} (Dementors: {fdp_dem})")
    
    # Combined Optimal Path (Extra Credit)
    combined_path = dijkstra_combined(graph_combined, province, 'Ottawa')
    if combined_path:
        hops = len(combined_path) - 1
        distance = calculate_total(graph, combined_path, 1)
        time_total = calculate_total(graph, combined_path, 2)
        dementors_total = calculate_total(graph, combined_path, 3)
        combined_score = calculate_combined_score(graph_combined, combined_path)
        
        print("== Extra Credit: Combined Optimal Path (All Criteria Minimized) ==")
        print(f"   Path: {' -> '.join(combined_path)}")
        print(f"   Hops: {hops}, Distance: {distance} km, Time: {time_total} hrs, Dementors: {dementors_total}")
        print(f"   Combined Score: {combined_score:.4f}")

# Visualize the graph
G = nx.DiGraph()
for edge in edges:
    G.add_edge(edge[0], edge[1], distance=edge[2], time=edge[3], dementors=edge[4])

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(15, 10))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'distance'), font_color='red')
plt.title("Magical Transportation Network (Distance in km)")
plt.savefig("magical_network.png", format="PNG")
plt.show()