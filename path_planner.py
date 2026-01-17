"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""

import numpy as np
from math import sqrt
import random

class PathPlannerBase:
    """Base class for all path planners"""
    def is_path_collision_free(self, p1, p2, num_checks=20):
        """
        Check if the line segment between p1 and p2 is collision-free
        """
        for i in range(num_checks + 1):
            t = i / num_checks
            point = (
                p1[0] + t * (p2[0] - p1[0]),
                p1[1] + t * (p2[1] - p1[1]),
                p1[2] + t * (p2[2] - p1[2])
            )
            if self.env.is_outside(point) or self.env.is_collide(point):
                return False
        return True

    def smooth_path(self, path, iterations=100, method='greedy'):
        if len(path) <= 2:
            return path
        if method == 'greedy':
            return self._smooth_greedy(path, iterations)
        elif method == 'random':
            return self._smooth_random(path, iterations)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    def _smooth_greedy(self, path, iterations):
        """Greedy deterministic smoothing"""
        smoothed = path.copy()      
        for _ in range(iterations):
            improved = False
            i = 0
            while i < len(smoothed) - 1:
                for j in range(len(smoothed) - 1, i + 1, -1):
                    if j - i <= 1:
                        break
                    
                    if self.is_path_collision_free(smoothed[i], smoothed[j]):
                        smoothed = np.vstack([smoothed[:i+1], smoothed[j:]])
                        improved = True
                        break
                i += 1
            if not improved:
                break
        
        return smoothed
    
    def _smooth_random(self, path, iterations):
        """Random sampling smoothing"""
        smoothed = path.copy()
        for _ in range(iterations):
            if len(smoothed) <= 2:
                break    
            i = random.randint(0, len(smoothed) - 3)
            j = random.randint(i + 2, len(smoothed) - 1)    
            if self.is_path_collision_free(smoothed[i], smoothed[j]):
                smoothed = np.vstack([smoothed[:i+1], smoothed[j:]])

        return smoothed
    
    def evaluate_path_quality(self, path):
        """
        Evaluate path quality with essential metrics
        
        Parameters:
            path: N×3 numpy array of waypoints
            
        Returns:
            dict: Dictionary containing quality metrics
                - total_length: total path length in meters
                - num_waypoints: number of waypoints
                - avg_segment_length: average distance between consecutive waypoints
                - smoothness: average curvature (lower is smoother)
                - max_turn_angle: maximum turning angle in degrees
        """
        if len(path) < 2:
            return {}
        
        # 1. Calculate path length
        segment_lengths = []
        for i in range(len(path) - 1):
            length = np.linalg.norm(path[i+1] - path[i])
            segment_lengths.append(length)
        
        total_length = np.sum(segment_lengths)
        
        # 2. Calculate turning angles (simpler and more intuitive than curvature)
        turn_angles = []
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-6 and v2_norm > 1e-6:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                turn_angles.append(np.degrees(angle))
        
        metrics = {
            'total_length': total_length,
            'num_waypoints': len(path),
            'avg_segment_length': np.mean(segment_lengths) if segment_lengths else 0,
            'avg_turn_angle': np.mean(turn_angles) if turn_angles else 0,
            'max_turn_angle': np.max(turn_angles) if turn_angles else 0,
        }
        
        return metrics
    
class AStarPlanner(PathPlannerBase):
    def __init__(self, env, resolution=0.5):
        """
        A* Path Planner
        
        Parameters:
            env: FlightEnvironment object
            resolution: grid resolution for discretization
        """
        self.env = env
        self.resolution = resolution
        
    def heuristic(self, pos1, pos2):
        """Euclidean distance heuristic"""
        return sqrt((pos1[0] - pos2[0])**2 + 
                   (pos1[1] - pos2[1])**2 + 
                   (pos1[2] - pos2[2])**2)
    
    def get_neighbors(self, node):
        """
        Get valid neighbors in 3D space (26-connectivity)
        """
        neighbors = []
        # 26 directions in 3D space
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    directions.append((dx, dy, dz))
        
        for dx, dy, dz in directions:
            new_pos = (
                node[0] + dx * self.resolution,
                node[1] + dy * self.resolution,
                node[2] + dz * self.resolution
            )
            
            # Check if the new position is valid
            if not self.env.is_outside(new_pos) and not self.env.is_collide(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from start to goal"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def plan(self, start, goal):
        """
        A* path planning algorithm
        
        Parameters:
            start: tuple (x, y, z) - start position
            goal: tuple (x, y, z) - goal position
            
        Returns:
            path: list of waypoints from start to goal
        """
        # Check if start and goal are valid
        if self.env.is_outside(start) or self.env.is_collide(start):
            raise ValueError("Start position is invalid!")
        if self.env.is_outside(goal) or self.env.is_collide(goal):
            raise ValueError("Goal position is invalid!")
        
        # Initialize open and closed sets
        open_set = {start}
        closed_set = set()
        
        # g_score: cost from start to node
        g_score = {start: 0}
        
        # f_score: estimated total cost from start to goal through node
        f_score = {start: self.heuristic(start, goal)}
        
        # came_from: for path reconstruction
        came_from = {}
        
        while open_set:
            # Get node with lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            # Check if we reached the goal
            if self.heuristic(current, goal) < self.resolution:
                path = self.reconstruct_path(came_from, current)
                if path[-1] != goal:
                    path.append(goal)
                return np.array(path)
            
            open_set.remove(current)
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + self.heuristic(current, neighbor)
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                
                # This path is the best so far
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
        
        # No path found
        raise ValueError("No valid path found from start to goal!")
    
class RRTPlanner(PathPlannerBase):
    def __init__(self, env, step_size=1.0, max_iterations=5000, goal_sample_rate=0.1):
        """
        RRT Path Planner
        
        Parameters:
            env: FlightEnvironment object
            step_size: maximum distance to extend the tree in each iteration
            max_iterations: maximum number of iterations
            goal_sample_rate: probability of sampling the goal (0-1)
        """
        self.env = env
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        
    class Node:
        """Tree node for RRT"""
        def __init__(self, position):
            self.position = np.array(position)
            self.parent = None
            self.cost = 0.0
    
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        return np.linalg.norm(pos1 - pos2)
    
    def sample_random_point(self, goal):
        """
        Sample a random point in the environment
        With probability goal_sample_rate, return the goal
        """
        if random.random() < self.goal_sample_rate:
            return np.array(goal)
        
        # Get environment bounds
        bounds = get_bounds(self.env)
        
        x = random.uniform(bounds[0][0], bounds[0][1])
        y = random.uniform(bounds[1][0], bounds[1][1])
        z = random.uniform(bounds[2][0], bounds[2][1])
        
        return np.array([x, y, z])
    
    def get_nearest_node(self, tree, point):
        """Find the nearest node in the tree to the given point"""
        min_dist = float('inf')
        nearest_node = None
        
        for node in tree:
            dist = self.distance(node.position, point)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        return nearest_node
    
    def steer(self, from_pos, to_pos):
        """
        Steer from from_pos towards to_pos with maximum step_size
        """
        from_pos = np.array(from_pos)
        to_pos = np.array(to_pos)
        
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_pos
        
        # Normalize and scale by step_size
        direction = direction / distance
        new_pos = from_pos + direction * self.step_size
        
        return new_pos
    
    def reconstruct_path(self, goal_node):
        """Reconstruct path from start to goal by backtracking through parents"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        path.reverse()
        return np.array(path)
    
    def plan(self, start, goal):
        """
        RRT path planning algorithm
        
        Parameters:
            start: tuple (x, y, z) - start position
            goal: tuple (x, y, z) - goal position
            
        Returns:
            path: numpy array of waypoints from start to goal
        """
        # Validate start and goal
        if self.env.is_outside(start) or self.env.is_collide(start):
            raise ValueError("Start position is invalid!")
        if self.env.is_outside(goal) or self.env.is_collide(goal):
            raise ValueError("Goal position is invalid!")
        
        # Initialize tree with start node
        start_node = self.Node(start)
        tree = [start_node]
        
        # Main RRT loop
        for iteration in range(self.max_iterations):
            # Sample random point
            random_point = self.sample_random_point(goal)
            
            # Find nearest node in tree
            nearest_node = self.get_nearest_node(tree, random_point)
            
            # Steer towards random point
            new_pos = self.steer(nearest_node.position, random_point)
            
            # Check if path to new position is collision-free
            if self.is_path_collision_free(nearest_node.position, new_pos):
                # Create new node
                new_node = self.Node(new_pos)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + self.distance(nearest_node.position, new_pos)
                tree.append(new_node)
                
                # Check if we reached the goal
                if self.distance(new_pos, goal) < self.step_size:
                    # Try to connect directly to goal
                    if self.is_path_collision_free(new_pos, goal):
                        goal_node = self.Node(goal)
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + self.distance(new_pos, goal)
                        
                        # Reconstruct and return path
                        path = self.reconstruct_path(goal_node)
                        print(f"RRT: Path found in {iteration + 1} iterations with {len(tree)} nodes")
                        return path
        
        # No path found within max iterations
        raise ValueError(f"No valid path found after {self.max_iterations} iterations!")
    
def get_bounds(self):
    """
    Get the boundaries of the flight environment.
    
    Returns:
        list: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """
    return [
        [0, self.env_width],   # x bounds: [0, 20.0]
        [0, self.env_length],  # y bounds: [0, 20.0]
        [0, self.env_height]   # z bounds: [0, 5.0]
    ]