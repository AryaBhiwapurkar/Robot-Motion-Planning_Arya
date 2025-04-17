import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- ENVIRONMENT SETUP ---

# Start and goal positions
start = np.array([1, 1])
goal = np.array([9, 9])
step_size = 0.2
threshold = 0.3  # distance to consider as 'reached goal'

# Obstacles (list of matplotlib.patches)
obstacles = [
    patches.Rectangle((3, 3), 2, 2),
    patches.Rectangle((6, 5), 2, 3)
]

# --- HELPER FUNCTIONS ---

def is_collision(point):
    """Check if the point is inside any obstacle"""
    for obs in obstacles:
        if obs.contains_point(point):
            return True
    return False

def follow_boundary(curr):
    """Simple boundary follow – move along the obstacle's edge clockwise"""
    # Naive logic: try small steps in clockwise direction
    directions = [
        [0, step_size], [step_size, 0], [0, -step_size], [-step_size, 0]
    ]
    for _ in range(100):  # limit to avoid infinite loops
        for d in directions:
            new = curr + d
            if not is_collision(new):
                return new
    return curr  # fallback

# --- BUG1 MAIN LOOP ---

path = [start.copy()]
curr = start.copy()
mode = 'go-to-goal'
closest_to_goal = curr
min_dist = np.linalg.norm(curr - goal)

while np.linalg.norm(curr - goal) > threshold and len(path) < 1000:
    if mode == 'go-to-goal':
        direction = goal - curr
        direction = direction / np.linalg.norm(direction)
        next_pos = curr + step_size * direction
        if is_collision(next_pos):
            mode = 'boundary-follow'
            closest_to_goal = curr
            min_dist = np.linalg.norm(curr - goal)
        else:
            curr = next_pos
            path.append(curr.copy())
    elif mode == 'boundary-follow':
        curr = follow_boundary(curr)
        path.append(curr.copy())
        dist = np.linalg.norm(curr - goal)
        if dist < min_dist:
            min_dist = dist
            closest_to_goal = curr
        # Check if we're back at closest point
        if np.allclose(curr, closest_to_goal, atol=0.1):
            mode = 'go-to-goal'

# --- PLOTTING ---

fig, ax = plt.subplots()
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_aspect('equal')
ax.plot(goal[0], goal[1], 'go', label='Goal')
ax.plot(start[0], start[1], 'ro', label='Start')

# Draw obstacles
for obs in obstacles:
    ax.add_patch(obs)

# Draw path
path = np.array(path)
ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')

plt.legend()
plt.title("Bug1 Algorithm Simulation")
plt.grid(True)
plt.show()
