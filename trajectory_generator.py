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

class QuinticPolynomialTrajectory(TrajectoryGenerator):

    def __init__(self, path, velocity=2.0):
        super().__init__(path, velocity)
    
    def quintic_polynomial_coefficients(self, points, times):
        """
        Compute quintic (5th order) polynomial coefficients
        ✅ 改进：使用更好的速度和加速度估计
        """
        n = len(points) - 1
        coeffs = []
        
        velocities = []
        for i in range(len(points)):
            if i == 0:
                # 起点：使用前向差分
                v = (points[i+1] - points[i]) / (times[i+1] - times[i])
            elif i == len(points) - 1:
                # 终点：使用后向差分
                v = (points[i] - points[i-1]) / (times[i] - times[i-1])
            else:
                # 中间点：使用中心差分（更平滑）
                dt_prev = times[i] - times[i-1]
                dt_next = times[i+1] - times[i]
                
                # 加权平均，考虑时间间隔
                w_prev = dt_next / (dt_prev + dt_next)
                w_next = dt_prev / (dt_prev + dt_next)
                
                v_prev = (points[i] - points[i-1]) / dt_prev
                v_next = (points[i+1] - points[i]) / dt_next
                
                v = w_prev * v_prev + w_next * v_next
            
            velocities.append(v)
        
        # 为每个段生成系数
        for i in range(n):
            t0 = times[i]
            t1 = times[i+1]
            dt = t1 - t0
            
            p0 = points[i]
            p1 = points[i+1]
            v0 = velocities[i]
            v1 = velocities[i+1]
            
            # ✅ 改进：估计加速度（使用速度的变化率）
            if i == 0:
                a0 = 0  # 起点加速度为0
            else:
                dt_prev = times[i] - times[i-1]
                a0 = (velocities[i] - velocities[i-1]) / dt_prev * 0.5  # 减小加速度突变
            
            if i == n - 1:
                a1 = 0  # 终点加速度为0
            else:
                dt_next = times[i+2] - times[i+1]
                a1 = (velocities[i+2] - velocities[i+1]) / dt_next * 0.5
            
            # 构建线性方程组求解quintic系数
            A = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, dt, dt**2, dt**3, dt**4, dt**5],
                [0, 1, 2*dt, 3*dt**2, 4*dt**3, 5*dt**4],
                [0, 0, 2, 6*dt, 12*dt**2, 20*dt**3]
            ])
            
            b = np.array([p0, v0, a0, p1, v1, a1])
            
            try:
                coeff = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # 如果求解失败，使用最小二乘
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            
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

def evaluate_trajectory_quality(trajectory, t_array, waypoints):
    """
    统一的轨迹质量评估函数
    专门用于对比 Cubic Spline vs Quintic Polynomial
    
    Returns:
        dict: Quality metrics
            - max_velocity: 最大速度 (m/s)
            - avg_velocity: 平均速度 (m/s)
            - max_acceleration: 最大加速度 (m/s²)
            - max_jerk: 最大 jerk (m/s³) - 关键对比指标
            - jerk_cost: jerk 积分 - 关键对比指标
            - max_waypoint_deviation: 最大航点偏差 (m)
    """
    trajectory = np.array(trajectory)
    t_array = np.array(t_array)
    waypoints = np.array(waypoints)
    
    dt = t_array[1] - t_array[0]
    
    # 计算导数
    velocity = np.gradient(trajectory, dt, axis=0)
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    
    acceleration = np.gradient(velocity, dt, axis=0)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    
    jerk = np.gradient(acceleration, dt, axis=0)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    
    # 计算航点偏差
    waypoint_deviations = []
    for waypoint in waypoints:
        distances = np.linalg.norm(trajectory - waypoint, axis=1)
        min_distance = np.min(distances)
        waypoint_deviations.append(min_distance)
    
    # 计算平滑度成本
    jerk_cost = np.sum(jerk_magnitude**2) * dt
    
    metrics = {
        'max_velocity': np.max(velocity_magnitude),
        'avg_velocity': np.mean(velocity_magnitude),
        'max_acceleration': np.max(acceleration_magnitude),
        'max_jerk': np.max(jerk_magnitude),
        'jerk_cost': jerk_cost,
        'max_waypoint_deviation': np.max(waypoint_deviations),
    }
    
    return metrics

def print_trajectory_metrics(metrics):
    print(f"\nQuinticPolynomialVelocity:")
    print(f"  Max:     {metrics['max_velocity']:.4f} m/s")
    print(f"  Average: {metrics['avg_velocity']:.4f} m/s")
    
    print(f"\nQuinticPolynomialAcceleration:")
    print(f"  Max:     {metrics['max_acceleration']:.4f} m/s²")
    
    print(f"\nQuinticPolynomialJerk (smoothness):")
    print(f"  Max:     {metrics['max_jerk']:.4f} m/s³")
    print(f"  Cost:    {metrics['jerk_cost']:.4f}")
    
    print(f"\nQuinticPolynomialWaypoint Deviation:")
    print(f"  Max:     {metrics['max_waypoint_deviation']:.4f} m")
    
    # 检查约束
    if metrics['max_waypoint_deviation'] > 0.1:
        print(f"  WARNING: Exceeds 0.1m limit!")
    else:
        print(f"  Within 0.1m limit")
    
    print(f"{'='*60}\n")
