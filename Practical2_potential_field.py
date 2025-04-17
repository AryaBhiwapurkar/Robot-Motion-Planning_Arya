import numpy as np
import matplotlib.pyplot as plt

# ---------- Parameters ----------
# These parameters control the behavior of the potential fields

# k_att: Attractive force constant (controls how strongly the robot is attracted to the goal)
k_att = 1.0

# k_rep: Repulsive force constant (controls how strongly the robot is repelled by obstacles)
k_rep = 15.0  # Moderate repulsion value

# rho_0: The threshold distance at which the repulsive force starts to decrease
rho_0 = 2.5

# alpha: A small constant that controls how far the robot moves each time based on the computed forces
alpha = 0.1

# random_perturbation: A small random noise added to the movement to help escape local minima
random_perturbation = 0.05

# start: The initial position of the robot in the 2D space
start = np.array([1.0, 1.0])

# goal: The target position the robot needs to reach
goal = np.array([8.0, 8.0])

# obstacles: A list of obstacles where each obstacle is represented by (x, y, radius)
# The robot must avoid these obstacles during its path planning
obstacles = [
    (4.0, 4.0, 1.0),  # Obstacle 1 (center: (4, 4), radius: 1.0)
    (6.0, 6.0, 1.2)   # Obstacle 2 (center: (6, 6), radius: 1.2)
]

# ---------- Force Functions ----------
# These functions calculate the attractive and repulsive forces

# Attractive Force (toward the goal)
def attractive_force(q, q_goal, k_att):
    # The attractive force pulls the robot toward the goal
    # The force is proportional to the vector pointing from the robot to the goal
    return -k_att * (q - q_goal)

# Repulsive Force (away from obstacles)
def repulsive_force(q, obstacles, k_rep, rho_0):
    force = np.zeros(2)  # Initialize a 2D zero vector to store the total repulsive force

    # Loop through each obstacle and calculate the repulsive force
    for obs in obstacles:
        center = np.array(obs[:2])  # Obstacle center (x, y)
        radius = obs[2]  # Obstacle radius
        dist = np.linalg.norm(q - center)  # Distance between robot and obstacle center
        rho = dist - radius  # Distance from obstacle boundary (radius)

        # If robot is inside the obstacle, it will experience a strong repulsive force
        if rho <= 0:
            direction = (q - center) / dist  # Calculate direction from obstacle to robot
            return k_rep * direction  # Apply repulsive force

        # If robot is within the influence range of the obstacle, apply a gradual repulsion
        elif rho < rho_0:
            direction = (q - center) / dist  # Calculate direction from obstacle to robot
            rep_val = k_rep * ((1/rho - 1/rho_0) / (rho**2))  # Calculate repulsive force magnitude
            force += rep_val * direction  # Add the repulsive force to total force
    return force  # Return the total repulsive force

# ---------- Simulation ----------
# This part handles the movement of the robot based on attractive and repulsive forces

position = start.copy()  # Initialize the robot's starting position
path = [position.copy()]  # Initialize the path with the starting position
prev_position = None  # This will hold the previous position, though not used in this code

# Simulation loop (runs for 1000 steps or until goal is reached)
for step in range(1000):
    # Calculate the attractive force toward the goal
    f_att = attractive_force(position, goal, k_att)
    
    # Calculate the repulsive force from the obstacles
    f_rep = repulsive_force(position, obstacles, k_rep, rho_0)
    
    # Total force is the sum of attractive and repulsive forces
    total_force = f_att + f_rep

    # Normalize the total force to avoid large movements
    if np.linalg.norm(total_force) > 0:
        total_force = total_force / np.linalg.norm(total_force)

    # Add a small random noise to the movement to escape local minima (situations where the robot gets stuck)
    noise = random_perturbation * (np.random.rand(2) - 0.5)

    # Calculate the new position based on the total force and the noise
    move = alpha * total_force + noise
    new_position = position + move

    # Append the new position to the path
    path.append(new_position.copy())
    
    # If the robot is close enough to the goal (within 0.2 units), stop the simulation
    if np.linalg.norm(new_position - goal) < 0.2:
        print("Goal reached.")  # Notify that the goal is reached
        break

    position = new_position  # Update the current position for the next iteration

# Convert the path list to a numpy array for easier plotting
path = np.array(path)

# ---------- Visualization ----------
# This part visualizes the path and obstacles using Matplotlib

# Create a figure with a size of 7x7 inches
plt.figure(figsize=(7, 7))

# Plot the robot's path as a blue line with dots
plt.plot(path[:, 0], path[:, 1], 'b.-', label='Robot Path')

# Plot the starting point (green circle)
plt.plot(start[0], start[1], 'go', label='Start')

# Plot the goal point (red circle)
plt.plot(goal[0], goal[1], 'ro', label='Goal')

# Plot each obstacle as a gray circle
for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.6)
    plt.gca().add_patch(circle)  # Add obstacle to the plot

# Set the title and labels
plt.title('Potential Field Path Planning with Obstacle Avoidance')
plt.grid(True)  # Enable grid for better visualization
plt.legend()  # Display the legend to label the start, goal, and path

# Set the limits of the plot to show the whole 2D space
plt.xlim(0, 10)
plt.ylim(0, 10)

# Ensure that the aspect ratio is equal (so the x and y axes have the same scale)
plt.gca().set_aspect('equal')

# Display the plot
plt.show()
