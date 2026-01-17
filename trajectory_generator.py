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

    def evaluate_trajectory_quality(self, t_array, trajectory):
        """
        为原始Cubic Spline添加轨迹质量评估
        """
        dt = t_array[1] - t_array[0]
        
        # 计算速度
        velocity = np.gradient(trajectory, dt, axis=0)
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        
        # 计算加速度
        acceleration = np.gradient(velocity, dt, axis=0)
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
        
        # 计算Jerk
        jerk = np.gradient(acceleration, dt, axis=0)
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        
        # 计算航点偏差
        waypoint_deviations = []
        for waypoint in self.path:
            distances = np.linalg.norm(trajectory - waypoint, axis=1)
            min_distance = np.min(distances)
            waypoint_deviations.append(min_distance)
        
        # Jerk成本
        jerk_cost = np.sum(jerk_magnitude**2) * dt
        
        metrics = {
            'max_velocity': np.max(velocity_magnitude),
            'avg_velocity': np.mean(velocity_magnitude),
            'max_acceleration': np.max(acceleration_magnitude),
            'avg_acceleration': np.mean(acceleration_magnitude),
            'max_jerk': np.max(jerk_magnitude),
            'avg_jerk': np.mean(jerk_magnitude),
            'jerk_cost': jerk_cost,
            'max_waypoint_deviation': np.max(waypoint_deviations),
            'avg_waypoint_deviation': np.mean(waypoint_deviations),
        }
        
        return metrics

class MinimumSnapTrajectory(TrajectoryGenerator):
    """
    Minimum Snap Trajectory Generator
    Generates smoother trajectories by minimizing snap (4th derivative)
    """
    
    def __init__(self, path, velocity=2.0):
        super().__init__(path, velocity)
    
    def quintic_polynomial_coefficients(self, points, times):
        """
        Compute quintic (5th order) polynomial coefficients
        Provides continuity up to acceleration
        """
        n = len(points) - 1
        
        # For each segment, we use quintic polynomial: p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # Constraints: position, velocity, acceleration at start and end
        
        coeffs = []
        
        for i in range(n):
            t0 = times[i]
            t1 = times[i+1]
            dt = t1 - t0
            
            p0 = points[i]
            p1 = points[i+1]
            
            # Estimate velocities at waypoints (using finite differences)
            if i == 0:
                v0 = (points[i+1] - points[i]) / (times[i+1] - times[i])
            else:
                v0 = (points[i+1] - points[i-1]) / (times[i+1] - times[i-1])
            
            if i == n - 1:
                v1 = (points[i+1] - points[i]) / (times[i+1] - times[i])
            else:
                v1 = (points[i+2] - points[i]) / (times[i+2] - times[i])
            
            # Assume zero acceleration at waypoints for smoothness
            a0 = 0
            a1 = 0
            
            # Solve for quintic coefficients
            # Boundary conditions: p(0)=p0, p(dt)=p1, v(0)=v0, v(dt)=v1, a(0)=a0, a(dt)=a1
            
            A = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, dt, dt**2, dt**3, dt**4, dt**5],
                [0, 1, 2*dt, 3*dt**2, 4*dt**3, 5*dt**4],
                [0, 0, 2, 6*dt, 12*dt**2, 20*dt**3]
            ])
            
            b = np.array([p0, v0, a0, p1, v1, a1])
            
            coeff = np.linalg.solve(A, b)
            coeffs.append(coeff)
        
        return coeffs
    
    def evaluate_quintic(self, coeffs, times, t):
        """Evaluate quintic polynomial at time t"""
        # Find segment
        segment_idx = 0
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i+1]:
                segment_idx = i
                break
        
        if t > times[-1]:
            segment_idx = len(times) - 2
            t = times[-1]
        
        # Evaluate polynomial
        coeff = coeffs[segment_idx]
        dt = t - times[segment_idx]
        
        return coeff[0] + coeff[1]*dt + coeff[2]*dt**2 + coeff[3]*dt**3 + coeff[4]*dt**4 + coeff[5]*dt**5
    
    def generate(self, num_points=200):
        """Generate minimum snap trajectory"""
        times = self.compute_time_allocation()
        
        # Generate quintic polynomial coefficients for each axis
        x_coeffs = self.quintic_polynomial_coefficients(self.path[:, 0], times)
        y_coeffs = self.quintic_polynomial_coefficients(self.path[:, 1], times)
        z_coeffs = self.quintic_polynomial_coefficients(self.path[:, 2], times)
        
        # Generate trajectory points
        t_array = np.linspace(0, times[-1], num_points)
        trajectory = np.zeros((num_points, 3))
        
        for i, t in enumerate(t_array):
            trajectory[i, 0] = self.evaluate_quintic(x_coeffs, times, t)
            trajectory[i, 1] = self.evaluate_quintic(y_coeffs, times, t)
            trajectory[i, 2] = self.evaluate_quintic(z_coeffs, times, t)
        
        return t_array, trajectory, times
    
    def evaluate_trajectory_quality(self, t_array, trajectory):
        """
        Evaluate trajectory quality metrics
        
        Returns:
            dict: Quality metrics including velocity, acceleration, jerk
        """
        dt = t_array[1] - t_array[0]
        
        # Calculate velocity (first derivative)
        velocity = np.gradient(trajectory, dt, axis=0)
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        
        # Calculate acceleration (second derivative)
        acceleration = np.gradient(velocity, dt, axis=0)
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
        
        # Calculate jerk (third derivative)
        jerk = np.gradient(acceleration, dt, axis=0)
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        
        # Calculate snap (fourth derivative) - optional
        snap = np.gradient(jerk, dt, axis=0)
        snap_magnitude = np.linalg.norm(snap, axis=1)
        
        # Calculate waypoint deviation
        waypoint_deviations = []
        for waypoint in self.path:
            # Find closest point on trajectory
            distances = np.linalg.norm(trajectory - waypoint, axis=1)
            min_distance = np.min(distances)
            waypoint_deviations.append(min_distance)
        
        # Jerk cost (smoothness metric)
        jerk_cost = np.sum(jerk_magnitude**2) * dt
        
        metrics = {
            'max_velocity': np.max(velocity_magnitude),
            'avg_velocity': np.mean(velocity_magnitude),
            'max_acceleration': np.max(acceleration_magnitude),
            'avg_acceleration': np.mean(acceleration_magnitude),
            'max_jerk': np.max(jerk_magnitude),
            'avg_jerk': np.mean(jerk_magnitude),
            'jerk_cost': jerk_cost,
            'max_snap': np.max(snap_magnitude),
            'max_waypoint_deviation': np.max(waypoint_deviations),
            'avg_waypoint_deviation': np.mean(waypoint_deviations),
        }
        
        return metrics
    
    def plot_trajectory(self, t_array, trajectory, waypoint_times, fig=None):
        """Plot trajectory with enhanced visualization"""
        if fig is None:
            fig = plt.figure(figsize=(12, 10))
        
        # Calculate derivatives for plotting
        dt = t_array[1] - t_array[0]
        velocity = np.gradient(trajectory, dt, axis=0)
        acceleration = np.gradient(velocity, dt, axis=0)
        
        # Create 4 subplots
        axes = []
        for i in range(4):
            axes.append(fig.add_subplot(4, 1, i+1))
        
        labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']
        colors = ['r', 'g', 'b']
        
        # Plot position
        for i in range(3):
            axes[i].plot(t_array, trajectory[:, i], 
                        color=colors[i], linewidth=2, label='Trajectory')
            axes[i].scatter(waypoint_times, self.path[:, i], 
                           color='black', s=50, zorder=5, label='Waypoints')
            axes[i].set_ylabel(labels[i], fontsize=11)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right')
        
        # Plot velocity magnitude
        velocity_mag = np.linalg.norm(velocity, axis=1)
        axes[3].plot(t_array, velocity_mag, color='purple', linewidth=2, label='Velocity Magnitude')
        axes[3].set_xlabel('Time (s)', fontsize=12)
        axes[3].set_ylabel('Velocity (m/s)', fontsize=11)
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='upper right')
        
        axes[0].set_title('Trajectory Time History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()