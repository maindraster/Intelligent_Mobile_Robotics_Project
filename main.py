from flight_environment import FlightEnvironment
from path_planner import AStarPlanner
from trajectory_generator import TrajectoryGenerator

env = FlightEnvironment(50)
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
raw_path = planner.plan(start, goal)
print(f"Raw path found with {len(raw_path)} waypoints")

# Smooth the path
path = planner.smooth_path(raw_path, iterations=50)
print(f"Smoothed path has {len(path)} waypoints")

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

print("Generating smooth trajectory...")
traj_gen = TrajectoryGenerator(path, velocity=2.0)
t_array, trajectory, waypoint_times = traj_gen.generate(num_points=200)
print(f"Trajectory generated with {len(trajectory)} points")
print(f"Total flight time: {t_array[-1]:.2f} seconds")

# Plot trajectory
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
