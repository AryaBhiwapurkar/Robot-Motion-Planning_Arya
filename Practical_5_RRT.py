import matplotlib.pyplot as plt
import random
import math

# --- Config ---
X_LIMIT = (0, 100)  # x-axis limits for the plot
Y_LIMIT = (0, 100)  # y-axis limits for the plot
STEP_SIZE = 5  # Step size when expanding a node towards the random point
GOAL_SAMPLE_RATE = 0.1  # 10% chance of sampling the goal point
MAX_ITER = 500  # Maximum number of iterations for the algorithm

# Define start and goal positions, as well as obstacles
start = (10, 10)  # Start point of the path
goal = (90, 90)  # Goal point
obstacles = [((40, 40), 10), ((60, 60), 15)]  # Obstacles as (center, radius) tuples


# --- Helpers ---
def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def is_in_obstacle(point):
    """Check if a point is inside any obstacle."""
    for (cx, cy), r in obstacles:
        if distance((cx, cy), point) <= r:  # Check if point is within any obstacle
            return True
    return False


class Node:
    """A Node in the RRT tree."""
    def __init__(self, point):
        self.point = point  # Coordinates of the node
        self.parent = None  # Parent node (for path reconstruction)


def get_random_point():
    """Randomly sample a point, with a 10% chance of sampling the goal."""
    if random.random() < GOAL_SAMPLE_RATE:  # 10% chance to sample the goal
        return goal
    return (random.randint(*X_LIMIT), random.randint(*Y_LIMIT))  # Otherwise, pick a random point


def steer(from_node, to_point):
    """Steer from a node towards a target point."""
    dist = distance(from_node.point, to_point)  # Calculate distance between current node and target
    if dist < STEP_SIZE:
        return to_point  # If the distance is less than step size, return the target
    # Calculate the direction and move towards the target with step size
    theta = math.atan2(to_point[1] - from_node.point[1], to_point[0] - from_node.point[0])
    new_point = (
        from_node.point[0] + STEP_SIZE * math.cos(theta),
        from_node.point[1] + STEP_SIZE * math.sin(theta)
    )
    return new_point


def is_collision_free(p1, p2):
    """Check if the path between two points is collision-free."""
    steps = int(distance(p1, p2) / 1)  # Divide the path into small steps for collision checking
    for i in range(steps):
        t = i / steps  # Interpolate between the two points
        x = p1[0] * (1 - t) + p2[0] * t
        y = p1[1] * (1 - t) + p2[1] * t
        if is_in_obstacle((x, y)):  # Check if any point along the path is inside an obstacle
            return False
    return True


# --- RRT Core ---
def rrt():
    """Implement the RRT algorithm."""
    nodes = [Node(start)]  # Initialize the tree with the start node

    for _ in range(MAX_ITER):
        rand_point = get_random_point()  # Sample a random point
        nearest = min(nodes, key=lambda node: distance(node.point, rand_point))  # Find the nearest node
        new_point = steer(nearest, rand_point)  # Steer from the nearest node towards the random point

        # If the new point is not in an obstacle and the path is collision-free, add the node to the tree
        if not is_in_obstacle(new_point) and is_collision_free(nearest.point, new_point):
            new_node = Node(new_point)
            new_node.parent = nearest  # Set parent node
            nodes.append(new_node)

            # If the new point is close enough to the goal, add the goal node and stop
            if distance(new_point, goal) < STEP_SIZE:
                final_node = Node(goal)
                final_node.parent = new_node
                nodes.append(final_node)
                print("Goal reached!")
                return nodes

    print("Max iterations reached.")
    return nodes


# --- Plotting ---
def draw_path(nodes):
    """Draw the tree and path on the plot."""
    plt.figure(figsize=(8, 8))
    
    # Plot obstacles as red circles
    for (cx, cy), r in obstacles:
        circle = plt.Circle((cx, cy), r, color='red')
        plt.gca().add_patch(circle)

    # Plot edges of the tree
    for node in nodes:
        if node.parent:
            x = [node.point[0], node.parent.point[0]]
            y = [node.point[1], node.parent.point[1]]
            plt.plot(x, y, '-b')  # Blue lines for the tree edges

    # Plot final path from the goal to the start
    goal_node = nodes[-1]
    path_x, path_y = [], []
    while goal_node:
        path_x.append(goal_node.point[0])  # Track the x-coordinates of the path
        path_y.append(goal_node.point[1])  # Track the y-coordinates of the path
        goal_node = goal_node.parent  # Move to the parent node
    plt.plot(path_x, path_y, '-g', linewidth=2)  # Green line for the final path

    # Plot start and goal points
    plt.plot(start[0], start[1], "go", markersize=10, label="Start")  # Green circle for the start
    plt.plot(goal[0], goal[1], "ro", markersize=10, label="Goal")  # Red circle for the goal
    
    # Set plot limits and grid
    plt.xlim(X_LIMIT)
    plt.ylim(Y_LIMIT)
    plt.grid()
    plt.legend()
    plt.title("RRT Path")
    plt.show()


# --- Run ---
tree = rrt()  # Run the RRT algorithm
draw_path(tree)  # Visualize the result
