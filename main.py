from flight_environment import FlightEnvironment
from path_planner import AStarPlanner, RRTPlanner, RRTStarPlanner
from trajectory_generator import TrajectoryGenerator, MinimumSnapTrajectory
import matplotlib.pyplot as plt

env = FlightEnvironment(100)
start = (1, 2, 0)
goal = (18, 18, 3)

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.

print("Starting A* path planning...")
planner = AStarPlanner(env, resolution=0.5)

# print("Starting RRT path planning...")
# planner = RRTPlanner(env, step_size=0.5)

# print("Starting RRT* path planning...")
# planner = RRTStarPlanner(env, step_size=0.5, rewire_radius=2.0,max_iterations=2000)

raw_path = planner.plan(start, goal)
print(f"Raw path found with {len(raw_path)} waypoints")

# Smooth the path
path = planner.smooth_path(raw_path, iterations=50)
print(f"Smoothed path has {len(path)} waypoints")

# 计算路径质量指标
path_metrics = planner.evaluate_path_quality(path)
print("\n=== Path Quality Metrics ===")
print(f"Total path length: {path_metrics['total_length']:.2f} m")
print(f"Number of waypoints: {path_metrics['num_waypoints']}")
print(f"Average segment length: {path_metrics['avg_segment_length']:.2f} m")
print(f"Path smoothness (avg curvature): {path_metrics['smoothness']:.4f}")
print(f"Max turning angle: {path_metrics['max_turn_angle']:.2f} degrees")

# --------------------------------------------------------------------------------------------------- #


env.plot_cylinders(path)


# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.

# print("Generating smooth trajectory...")
# traj_gen = TrajectoryGenerator(path, velocity=2.0)

print("\n=== Generating Minimum Snap Trajectory ===")
traj_gen = MinimumSnapTrajectory(path, velocity=2.0)
t_array, trajectory, waypoint_times = traj_gen.generate(num_points=200)
print(f"Trajectory generated with {len(trajectory)} points")
print(f"Total flight time: {t_array[-1]:.2f} seconds")

# 计算轨迹质量指标
traj_metrics = traj_gen.evaluate_trajectory_quality(t_array, trajectory)
print("\n=== Trajectory Quality Metrics ===")
print(f"Max velocity: {traj_metrics['max_velocity']:.2f} m/s")
print(f"Avg velocity: {traj_metrics['avg_velocity']:.2f} m/s")
print(f"Max acceleration: {traj_metrics['max_acceleration']:.2f} m/s²")
print(f"Avg acceleration: {traj_metrics['avg_acceleration']:.2f} m/s²")
print(f"Max jerk: {traj_metrics['max_jerk']:.2f} m/s³")
print(f"Trajectory smoothness (jerk integral): {traj_metrics['jerk_cost']:.2f}")
print(f"Waypoint deviation (max): {traj_metrics['max_waypoint_deviation']:.4f} m")
print(f"Waypoint deviation (avg): {traj_metrics['avg_waypoint_deviation']:.4f} m")

traj_gen.plot_trajectory(t_array, trajectory, waypoint_times)

# --------------------------------------------------------------------------------------------------- #

# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
