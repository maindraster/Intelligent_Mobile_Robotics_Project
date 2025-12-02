"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""


import numpy as np
import matplotlib.pyplot as plt


class TrajectoryGenerator:
    def __init__(self, path, velocity=2.0):
        """
        Generate smooth trajectory using cubic spline interpolation
        
        Parameters:
            path: N×3 numpy array of waypoints
            velocity: average velocity in m/s
        """
        self.path = np.array(path)
        self.velocity = velocity
        
    def compute_path_length(self):
        """Compute total path length"""
        total_length = 0
        for i in range(len(self.path) - 1):
            segment_length = np.linalg.norm(self.path[i+1] - self.path[i])
            total_length += segment_length
        return total_length
    
    def compute_time_allocation(self):
        """
        Allocate time for each segment based on distance
        """
        segment_lengths = []
        for i in range(len(self.path) - 1):
            length = np.linalg.norm(self.path[i+1] - self.path[i])
            segment_lengths.append(length)
        
        # Time allocation based on segment length
        times = [0]
        for length in segment_lengths:
            times.append(times[-1] + length / self.velocity)
        
        return np.array(times)
    
    def cubic_spline_coefficients(self, points, times):
        """
        Compute cubic spline coefficients for 1D interpolation
        Natural boundary conditions (second derivative = 0 at endpoints)
        """
        n = len(points) - 1
        h = np.diff(times)
        
        # Build tridiagonal system for second derivatives
        A = np.zeros((n+1, n+1))
        b = np.zeros(n+1)
        
        # Natural boundary conditions
        A[0, 0] = 1
        A[n, n] = 1
        
        # Interior points
        for i in range(1, n):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = 3 * ((points[i+1] - points[i]) / h[i] - 
                       (points[i] - points[i-1]) / h[i-1])
        
        # Solve for second derivatives
        M = np.linalg.solve(A, b)
        
        # Compute coefficients for each segment
        coeffs = []
        for i in range(n):
            a = points[i]
            b_coef = (points[i+1] - points[i]) / h[i] - h[i] * (2*M[i] + M[i+1]) / 3
            c = M[i]
            d = (M[i+1] - M[i]) / (3 * h[i])
            coeffs.append((a, b_coef, c, d))
        
        return coeffs
    
    def evaluate_spline(self, coeffs, times, t):
        """
        Evaluate cubic spline at time t
        """
        # Find the segment
        segment_idx = 0
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i+1]:
                segment_idx = i
                break
        
        if t > times[-1]:
            segment_idx = len(times) - 2
            t = times[-1]
        
        # Evaluate polynomial
        a, b, c, d = coeffs[segment_idx]
        dt = t - times[segment_idx]
        
        return a + b*dt + c*dt**2 + d*dt**3
    
    def generate(self, num_points=200):
        """
        Generate smooth trajectory
        
        Returns:
            t_array: time array
            trajectory: N×3 array of trajectory points
        """
        times = self.compute_time_allocation()
        
        # Generate spline coefficients for each axis
        x_coeffs = self.cubic_spline_coefficients(self.path[:, 0], times)
        y_coeffs = self.cubic_spline_coefficients(self.path[:, 1], times)
        z_coeffs = self.cubic_spline_coefficients(self.path[:, 2], times)
        
        # Generate trajectory points
        t_array = np.linspace(0, times[-1], num_points)
        trajectory = np.zeros((num_points, 3))
        
        for i, t in enumerate(t_array):
            trajectory[i, 0] = self.evaluate_spline(x_coeffs, times, t)
            trajectory[i, 1] = self.evaluate_spline(y_coeffs, times, t)
            trajectory[i, 2] = self.evaluate_spline(z_coeffs, times, t)
        
        return t_array, trajectory, times
    
    def plot_trajectory(self, t_array, trajectory, waypoint_times):
        """
        Plot trajectory with waypoints
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']
        colors = ['r', 'g', 'b']
        
        for i in range(3):
            # Plot continuous trajectory
            axes[i].plot(t_array, trajectory[:, i], 
                        color=colors[i], linewidth=2, label='Trajectory')
            
            # Plot discrete waypoints
            axes[i].scatter(waypoint_times, self.path[:, i], 
                           color='black', s=50, zorder=5, label='Waypoints')
            
            axes[i].set_xlabel('Time (s)', fontsize=12)
            axes[i].set_ylabel(labels[i], fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()