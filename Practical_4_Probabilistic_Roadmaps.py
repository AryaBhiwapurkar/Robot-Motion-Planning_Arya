import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import math

# --- Config ---
# This section defines all the necessary parameters for the PRM algorithm

NUM_SAMPLES = 200  # Number of free space samples to generate
K = 10  # Number of nearest neighbors for each sample point to connect
X_LIMIT = (0, 100)  # X-coordinate range for the free space
Y_LIMIT = (0, 100)  # Y-coordinate range for the free space
start = (10, 10)  # Start point coordinates
goal = (90, 90)   # Goal point coordinates
# List of obstacles, each represented by a center (x, y) and radius
obstacles = [((40, 40), 10), ((60, 60), 15)]


# --- Collision Checking ---
# These functions are used to check if the robot's path is clear of obstacles

# Distance between two points in the 2D plane
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Check if a given point is inside any obstacle
def is_in_obstacle(point):
    for (cx, cy), r in obstacles:
        # Calculate the distance between the point and the obstacle center
        if distance(point, (cx, cy)) <= r:
            return True  # Point is inside the obstacle
    return False  # Point is not inside any obstacle

# Check if the line segment from p1 to p2 is free of collisions
def is_collision_free(p1, p2):
    # Get the number of steps to break the segment into (roughly the distance between p1 and p2)
    steps = int(distance(p1, p2))
    for i in range(steps + 1):
        t = i / steps if steps else 0  # Parametric interpolation parameter
        # Interpolate between p1 and p2 to generate points along the line segment
        x = p1[0] * (1 - t) + p2[0] * t
        y = p1[1] * (1 - t) + p2[1] * t
        if is_in_obstacle((x, y)):  # Check if any point along the line is in an obstacle
            return False  # Collision detected
    return True  # No collision detected


# --- Sample points ---
# This function generates a set of random points in free space (outside obstacles)

def sample_free_points():
    points = []
    while len(points) < NUM_SAMPLES:
        # Generate a random point within the specified limits
        x = random.uniform(*X_LIMIT)
        y = random.uniform(*Y_LIMIT)
        # Check if the point is not inside any obstacle
        if not is_in_obstacle((x, y)):
            points.append((x, y))  # Add the valid point to the list
    return points


# --- PRM Planner ---
# This part handles the construction of the probabilistic roadmap (PRM)

def build_prm():
    # Generate random points in free space
    points = sample_free_points()
    # Add the start and goal points to the list of points
    points.append(start)
    points.append(goal)

    # Create an undirected graph to represent the roadmap
    G = nx.Graph()

    # Add nodes to the graph, each node represents a free space point
    for i, p in enumerate(points):
        G.add_node(i, pos=p)

    # For each point, find the K nearest neighbors and check for collision-free edges
    for i, p1 in enumerate(points):
        # Find the distances from p1 to all other points
        distances = [(j, distance(p1, p2)) for j, p2 in enumerate(points) if j != i]
        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[1])
        # Add edges to the graph for the K nearest neighbors, if the path is collision-free
        for j, _ in distances[:K]:
            p2 = points[j]
            if is_collision_free(p1, p2):
                # Add an edge with the distance as the edge weight (representing cost)
                G.add_edge(i, j, weight=distance(p1, p2))

    return G, points


# Find the shortest path from start to goal using the constructed graph
def find_path(G, points):
    start_idx = len(points) - 2  # Index of the start point in the points list
    goal_idx = len(points) - 1   # Index of the goal point in the points list
    try:
        # Use NetworkX to find the shortest path based on edge weights (distance)
        path = nx.shortest_path(G, start_idx, goal_idx, weight='weight')
        return [points[i] for i in path]  # Return the list of points on the path
    except nx.NetworkXNoPath:
        # If no path is found, print a message and return an empty path
        print("No path found")
        return []


# --- Visualization ---
# This function visualizes the PRM graph and the shortest path

def draw(prm_graph, points, path):
    plt.figure(figsize=(8, 8))  # Create a new figure for plotting

    # Draw obstacles as red circles
    for (cx, cy), r in obstacles:
        circle = plt.Circle((cx, cy), r, color='red')
        plt.gca().add_patch(circle)

    # Draw edges in the PRM graph
    for u, v in prm_graph.edges:
        x = [points[u][0], points[v][0]]  # X coordinates of the edge endpoints
        y = [points[u][1], points[v][1]]  # Y coordinates of the edge endpoints
        plt.plot(x, y, 'skyblue', linewidth=0.5)  # Plot edges in sky blue

    # If a path is found, plot it in green
    if path:
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        plt.plot(x, y, 'g', linewidth=2)  # Plot the path in green

    # Plot the start and goal points as green and red circles
    plt.plot(start[0], start[1], "go", markersize=10, label="Start")
    plt.plot(goal[0], goal[1], "ro", markersize=10, label="Goal")

    # Set the plot limits and labels
    plt.xlim(X_LIMIT)
    plt.ylim(Y_LIMIT)
    plt.title("Probabilistic Roadmap (PRM)")
    plt.grid()  # Display a grid
    plt.legend()  # Display the legend
    plt.show()  # Show the plot



# This part executes the PRM algorithm and visualizes the results

prm_graph, points = build_prm()  # Build the PRM graph and get the points
path = find_path(prm_graph, points)  # Find the shortest path from start to goal
draw(prm_graph, points, path)  # Visualize the PRM graph and the path
