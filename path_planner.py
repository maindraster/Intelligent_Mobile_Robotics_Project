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


class AStarPlanner:
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
    
    def smooth_path(self, path, iterations=50):
        """
        Smooth the path by removing unnecessary waypoints
        """
        if len(path) <= 2:
            return path
        
        # Convert to list of tuples for easier handling
        smoothed = [tuple(p) for p in path]
        
        for _ in range(iterations):
            i = 0
            while i < len(smoothed) - 1:
                # Try to connect current point to points further ahead
                for j in range(len(smoothed) - 1, i + 1, -1):
                    if self.is_line_collision_free(smoothed[i], smoothed[j]):
                        # Remove intermediate points
                        smoothed = smoothed[:i+1] + smoothed[j:]
                        break
                i += 1
        
        # Check if last point matches goal
        last_point = np.array(smoothed[-1])
        goal_point = np.array(path[-1])
        if not np.allclose(last_point, goal_point):
            smoothed.append(tuple(path[-1]))
        
        return np.array(smoothed)
    
    def is_line_collision_free(self, p1, p2, num_checks=20):
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
