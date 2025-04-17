import numpy as np
import matplotlib.pyplot as plt

# --- Initialization ---
dt = 1.0  # Time step (1 second)

# Initial state: [x, y, vx, vy] where x and y are position, vx and vy are velocity
x = np.array([[0], [0], [1], [1]])  # Initial position at (0, 0), velocity (1, 1)

# State transition matrix (F): Describes how the state evolves from the previous step.
# Each row describes how each component of the state affects others.
F = np.array([
    [1, 0, dt, 0],  # x_new = x + vx * dt
    [0, 1, 0, dt],  # y_new = y + vy * dt
    [0, 0, 1,  0],  # vx_new = vx (no change in velocity)
    [0, 0, 0,  1]   # vy_new = vy (no change in velocity)
])

# Measurement matrix (H): We only observe the position (x, y), not velocity.
H = np.array([
    [1, 0, 0, 0],  # Measure x position directly
    [0, 1, 0, 0]   # Measure y position directly
])

# Initial uncertainty (P): A large value indicates we are uncertain about the initial state.
P = np.eye(4) * 500  # Large uncertainty in initial state

# Measurement noise (R): This matrix represents the noise in sensor measurements, like GPS noise.
R = np.eye(2) * 5  # Measurement noise of 5 units in x and y directions

# Process noise (Q): This matrix represents model uncertainty, which accounts for errors in the system's prediction.
Q = np.eye(4) * 0.2  # Small process noise for state prediction

# Identity matrix (I): Used for matrix calculations in the Kalman filter update step.
I = np.eye(4)

# Simulated sensor data for visualization
true_path = []  # True path of the object
measured_path = []  # Noisy measured positions
estimated_path = []  # Kalman filter's estimated positions

# Kalman Filter loop
for i in range(50):
    # --- Simulate true position ---
    true_x = i * dt  # True x position: moves at 1 unit per second
    true_y = i * dt  # True y position: moves at 1 unit per second
    z_true = np.array([[true_x], [true_y]])  # True position (x, y)
    
    # Simulated noisy measurement (sensor noise added to true position)
    z = z_true + np.random.randn(2, 1) * 2.5  # Random noise with standard deviation of 2.5

    # --- Prediction Step ---
    x = F @ x  # Predict next state based on previous state (state transition)
    P = F @ P @ F.T + Q  # Update uncertainty based on the state transition model and process noise

    # --- Update Step ---
    y = z - H @ x  # Measurement residual (difference between predicted and measured)
    S = H @ P @ H.T + R  # Residual covariance (uncertainty of the residual)
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain (how much to adjust the prediction based on the measurement)

    x = x + K @ y  # Update state estimate (use Kalman gain to adjust prediction)
    P = (I - K @ H) @ P  # Update uncertainty (reduce uncertainty based on Kalman gain)

    # Store for visualization
    true_path.append(z_true.flatten())  # True path (without noise)
    measured_path.append(z.flatten())  # Measured noisy positions
    estimated_path.append(x[:2].flatten())  # Estimated path (Kalman filter output)

# --- Plot ---
true_path = np.array(true_path)
measured_path = np.array(measured_path)
estimated_path = np.array(estimated_path)

# Create a plot to visualize the paths
plt.figure(figsize=(8, 6))
plt.plot(true_path[:,0], true_path[:,1], 'g--', label="True Path")  # True path (green dashed line)
plt.plot(measured_path[:,0], measured_path[:,1], 'rx', label="Measured")  # Measured noisy positions (red crosses)
plt.plot(estimated_path[:,0], estimated_path[:,1], 'b-', label="Kalman Estimated")  # Estimated path (blue line)
plt.legend()  # Show legend
plt.grid(True)  # Show grid
plt.title("Kalman Filter for 2D Localization")  # Title of the plot
plt.xlabel("X")  # Label for X-axis
plt.ylabel("Y")  # Label for Y-axis
plt.axis("equal")  # Equal scaling for both axes
plt.show()

# ------------------------------------------------------------
# 📈 Kalman Filter Visualization - 2D Localization
#
# This plot helps visualize the performance of the Kalman Filter
# in estimating a robot's position in 2D space.
#
# ➤ Green Dashed Line (True Path):
#     - The actual path the robot is expected to follow.
#     - Used for validation in simulation; not available in real-world scenarios.
#
# ➤ Red Crosses (Measured Positions):
#     - Noisy measurements simulating real-world sensor inputs (e.g., GPS).
#     - Show how far off raw data can be from the true location.
#
# ➤ Blue Line (Kalman Estimated Path):
#     - The output of the Kalman Filter.
#     - This is a smoothed estimate based on previous state + current measurement.
#     - It tends to stay close to the true path while ignoring measurement noise.
#
# ✅ Interpretation:
#     - The Kalman Filter effectively reduces measurement noise.
#     - Estimated path is more reliable than raw sensor readings.
#     - Used in robotics to improve localization accuracy in uncertain environments.
# ------------------------------------------------------------
