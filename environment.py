import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
import json
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
import time

class MultiRobotEnvironment:
    """
    Standalone Multi-Robot Environment for Path Planning Testing
    
    This environment can be used with any algorithm (Q-learning, DQN, A*, Hungarian, etc.)
    by implementing the action interface.
    """
    
    def __init__(self, grid_size=(15, 15), num_robots=3, num_tasks=5, seed=None):
        """
        Initialize the multi-robot environment
        
        Args:
            grid_size: Tuple of (rows, cols) for the grid
            num_robots: Number of robots in the environment
            num_tasks: Number of tasks (pickup-dropoff pairs)
            seed: Random seed for reproducible experiments
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.max_steps = 300  # Maximum steps per episode
        
        # Environment state
        self.robots = {}
        self.tasks = {}
        self.static_obstacles = set()
        self.dynamic_obstacles = {}
        self.completed_tasks = set()
        self.step_count = 0
        self.collision_count = 0
        self.total_distance_traveled = 0
        
        # Actions: 0=stay, 1=up, 2=right, 3=down, 4=left
        self.actions = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['Stay', 'Up', 'Right', 'Down', 'Left']
        
        # History for visualization
        self.position_history = []
        self.collision_history = []
        
        self.reset()
        
        print("🏭 Multi-Robot Environment Initialized")
        print(f"   Grid Size: {self.grid_size}")
        print(f"   Robots: {self.num_robots}")
        print(f"   Tasks: {self.num_tasks}")
        print(f"   Max Steps: {self.max_steps}")
    
    def reset(self):
        """Reset environment to initial state"""
        self.step_count = 0
        self.collision_count = 0
        self.total_distance_traveled = 0
        self.completed_tasks = set()
        self.position_history = []
        self.collision_history = []
        
        # Generate static obstacles (walls, fixed machinery)
        self._generate_static_obstacles()
        
        # Initialize robots at random valid positions
        self.robots = {}
        occupied_positions = set()
        
        for i in range(self.num_robots):
            while True:
                pos = (random.randint(0, self.grid_size[0]-1), 
                       random.randint(0, self.grid_size[1]-1))
                if (pos not in self.static_obstacles and 
                    pos not in occupied_positions):
                    self.robots[i] = {
                        'position': pos,
                        'assigned_task': None,
                        'carrying_task': None,
                        'idle_time': 0,
                        'total_distance': 0,
                        'path_history': [pos]
                    }
                    occupied_positions.add(pos)
                    break
        
        # Generate tasks (pickup-dropoff pairs)
        self.tasks = {}
        for i in range(self.num_tasks):
            while True:
                pickup = (random.randint(0, self.grid_size[0]-1), 
                         random.randint(0, self.grid_size[1]-1))
                dropoff = (random.randint(0, self.grid_size[0]-1), 
                          random.randint(0, self.grid_size[1]-1))
                
                # Ensure task locations are valid and not occupied
                all_occupied = (occupied_positions | 
                               self.static_obstacles |
                               set([t['pickup'] for t in self.tasks.values()]) |
                               set([t['dropoff'] for t in self.tasks.values()]))
                
                if (pickup not in all_occupied and 
                    dropoff not in all_occupied and 
                    pickup != dropoff):
                    self.tasks[i] = {
                        'pickup': pickup,
                        'dropoff': dropoff,
                        'completed': False,
                        'assigned_to': None,
                        'priority': random.uniform(0.5, 2.0)  # Task priority
                    }
                    break
        
        # Generate dynamic obstacles (moving machinery)
        self._update_dynamic_obstacles()
        
        # Store initial state
        self.position_history.append(self._get_positions_snapshot())
        
        return self.get_state()
    
    def _generate_static_obstacles(self):
        """Generate static obstacles in the environment"""
        self.static_obstacles = set()
        
        # Create some wall-like obstacles
        num_walls = random.randint(2, 4)
        for _ in range(num_walls):
            # Random wall direction (horizontal or vertical)
            if random.random() < 0.5:
                # Horizontal wall
                row = random.randint(1, self.grid_size[0]-2)
                start_col = random.randint(0, self.grid_size[1]-4)
                length = random.randint(3, min(6, self.grid_size[1] - start_col))
                for col in range(start_col, start_col + length):
                    self.static_obstacles.add((row, col))
            else:
                # Vertical wall
                col = random.randint(1, self.grid_size[1]-2)
                start_row = random.randint(0, self.grid_size[0]-4)
                length = random.randint(3, min(6, self.grid_size[0] - start_row))
                for row in range(start_row, start_row + length):
                    self.static_obstacles.add((row, col))
        
        # Add some scattered obstacles
        num_scattered = random.randint(3, 8)
        for _ in range(num_scattered):
            while True:
                pos = (random.randint(0, self.grid_size[0]-1), 
                       random.randint(0, self.grid_size[1]-1))
                if pos not in self.static_obstacles:
                    self.static_obstacles.add(pos)
                    break
    
    def _update_dynamic_obstacles(self):
        """Update positions of dynamic obstacles (moving machinery)"""
        # Keep some existing dynamic obstacles and move them
        new_dynamic = {}
        for obs_id, old_pos in self.dynamic_obstacles.items():
            # 70% chance to keep moving, 30% chance to stop
            if random.random() < 0.7:
                # Try to move in a random direction
                moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
                for move in random.sample(moves, len(moves)):
                    new_pos = (old_pos[0] + move[0], old_pos[1] + move[1])
                    if (0 <= new_pos[0] < self.grid_size[0] and 
                        0 <= new_pos[1] < self.grid_size[1] and
                        new_pos not in self.static_obstacles):
                        new_dynamic[obs_id] = new_pos
                        break
                else:
                    new_dynamic[obs_id] = old_pos  # Stay in place if can't move
        
        # Add new dynamic obstacles occasionally
        if len(new_dynamic) < 3 and random.random() < 0.3:
            attempts = 10
            while attempts > 0:
                pos = (random.randint(0, self.grid_size[0]-1), 
                       random.randint(0, self.grid_size[1]-1))
                if (pos not in self.static_obstacles and 
                    pos not in new_dynamic.values() and
                    pos not in [r['position'] for r in self.robots.values()]):
                    new_id = max(new_dynamic.keys(), default=-1) + 1
                    new_dynamic[new_id] = pos
                    break
                attempts -= 1
        
        self.dynamic_obstacles = new_dynamic
    
    def get_state(self):
        """Get current complete state of the environment"""
        return {
            'robots': {k: v.copy() for k, v in self.robots.items()},
            'tasks': {k: v.copy() for k, v in self.tasks.items()},
            'static_obstacles': self.static_obstacles.copy(),
            'dynamic_obstacles': self.dynamic_obstacles.copy(),
            'step_count': self.step_count,
            'completed_tasks': self.completed_tasks.copy(),
            'grid_size': self.grid_size,
            'collision_count': self.collision_count
        }
    
    def get_robot_observation(self, robot_id):
        """Get observation for a specific robot (partial observability)"""
        if robot_id not in self.robots:
            return None
        
        robot = self.robots[robot_id]
        robot_pos = robot['position']
        
        # Observable range (5x5 area around robot)
        obs_range = 2
        local_obstacles = []
        local_robots = []
        
        for dx in range(-obs_range, obs_range + 1):
            for dy in range(-obs_range, obs_range + 1):
                check_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
                if (0 <= check_pos[0] < self.grid_size[0] and 
                    0 <= check_pos[1] < self.grid_size[1]):
                    
                    # Check for obstacles
                    if (check_pos in self.static_obstacles or 
                        check_pos in self.dynamic_obstacles.values()):
                        local_obstacles.append((dx, dy))
                    
                    # Check for other robots
                    for other_id, other_robot in self.robots.items():
                        if other_id != robot_id and other_robot['position'] == check_pos:
                            local_robots.append((dx, dy, other_id))
        
        # Get assigned task info
        task_info = None
        if robot['assigned_task'] is not None:
            task = self.tasks[robot['assigned_task']]
            target = task['pickup'] if robot['carrying_task'] is None else task['dropoff']
            task_info = {
                'target': target,
                'distance': self._manhattan_distance(robot_pos, target),
                'carrying': robot['carrying_task'] is not None
            }
        
        return {
            'position': robot_pos,
            'local_obstacles': local_obstacles,
            'local_robots': local_robots,
            'task_info': task_info,
            'grid_size': self.grid_size
        }
    
    def step(self, actions_dict):
        """
        Execute one step in the environment
        
        Args:
            actions_dict: Dict {robot_id: action} where action is 0-4
                         0=stay, 1=up, 2=right, 3=down, 4=left
        
        Returns:
            tuple: (next_state, rewards_dict, done, info)
        """
        self.step_count += 1
        
        # Store old positions for collision detection and distance calculation
        old_positions = {robot_id: robot['position'] for robot_id, robot in self.robots.items()}
        
        # Execute actions for all robots
        for robot_id, action in actions_dict.items():
            if robot_id in self.robots and 0 <= action <= 4:
                self._execute_robot_action(robot_id, action)
        
        # Update dynamic obstacles every 8 steps
        if self.step_count % 8 == 0:
            self._update_dynamic_obstacles()
        
        # Check for collisions
        collision_occurred = self._check_collisions(old_positions)
        if collision_occurred:
            self.collision_history.append(self.step_count)
        
        # Handle task assignments and completions
        self._handle_tasks()
        
        # Calculate rewards
        rewards = self._calculate_rewards(old_positions)
        
        # Update position history
        self.position_history.append(self._get_positions_snapshot())
        
        # Check if episode is done
        done = self._check_done()
        
        # Additional info
        info = {
            'collisions_this_step': collision_occurred,
            'tasks_completed': len(self.completed_tasks),
            'total_tasks': self.num_tasks
        }
        
        return self.get_state(), rewards, done, info
    
    def _execute_robot_action(self, robot_id, action):
        """Execute action for a specific robot"""
        robot = self.robots[robot_id]
        old_pos = robot['position']
        
        # Get new position based on action
        dx, dy = self.actions[action]
        new_x = max(0, min(self.grid_size[0]-1, old_pos[0] + dx))
        new_y = max(0, min(self.grid_size[1]-1, old_pos[1] + dy))
        new_pos = (new_x, new_y)
        
        # Check if new position is valid (not an obstacle)
        if (new_pos not in self.static_obstacles and 
            new_pos not in self.dynamic_obstacles.values()):
            robot['position'] = new_pos
            robot['path_history'].append(new_pos)
            
            # Calculate distance traveled
            if new_pos != old_pos:
                distance = np.sqrt((new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2)
                robot['total_distance'] += distance
                self.total_distance_traveled += distance
                robot['idle_time'] = 0  # Reset idle time when moving
            else:
                robot['idle_time'] += 1
        else:
            # Can't move, increase idle time
            robot['idle_time'] += 1
    
    def _check_collisions(self, old_positions):
        """Check for robot-robot collisions"""
        collision_occurred = False
        current_positions = [robot['position'] for robot in self.robots.values()]
        
        # Check for multiple robots at same position
        if len(current_positions) != len(set(current_positions)):
            self.collision_count += 1
            collision_occurred = True
        
        # Check for robots swapping positions
        robot_ids = list(self.robots.keys())
        for i in range(len(robot_ids)):
            for j in range(i+1, len(robot_ids)):
                robot1_id, robot2_id = robot_ids[i], robot_ids[j]
                old_pos1, old_pos2 = old_positions[robot1_id], old_positions[robot2_id]
                new_pos1, new_pos2 = self.robots[robot1_id]['position'], self.robots[robot2_id]['position']
                
                if old_pos1 == new_pos2 and old_pos2 == new_pos1:
                    self.collision_count += 1
                    collision_occurred = True
        
        return collision_occurred
    
    def _handle_tasks(self):
        """Handle task assignments and completions"""
        # Assign tasks to robots that don't have tasks
        for robot_id, robot in self.robots.items():
            if robot['assigned_task'] is None and robot['carrying_task'] is None:
                # Find unassigned tasks
                available_tasks = [tid for tid, task in self.tasks.items() 
                                 if not task['completed'] and task['assigned_to'] is None]
                
                if available_tasks:
                    robot_pos = robot['position']
                    # Choose task based on distance and priority
                    def task_score(tid):
                        task = self.tasks[tid]
                        distance = self._manhattan_distance(robot_pos, task['pickup'])
                        return task['priority'] / (distance + 1)  # Higher priority, lower distance = higher score
                    
                    best_task = max(available_tasks, key=task_score)
                    robot['assigned_task'] = best_task
                    self.tasks[best_task]['assigned_to'] = robot_id
        
        # Check if robots reached pickup locations
        for robot_id, robot in self.robots.items():
            if robot['assigned_task'] is not None and robot['carrying_task'] is None:
                task = self.tasks[robot['assigned_task']]
                if robot['position'] == task['pickup']:
                    robot['carrying_task'] = robot['assigned_task']
                    robot['assigned_task'] = None
        
        # Check if robots reached dropoff locations
        for robot_id, robot in self.robots.items():
            if robot['carrying_task'] is not None:
                task = self.tasks[robot['carrying_task']]
                if robot['position'] == task['dropoff']:
                    task['completed'] = True
                    self.completed_tasks.add(robot['carrying_task'])
                    robot['carrying_task'] = None
    
    def _calculate_rewards(self, old_positions):
        """Calculate rewards for each robot"""
        rewards = {}
        
        for robot_id, robot in self.robots.items():
            reward = 0
            
            # Large reward for completing a task
            if robot_id in old_positions:  # Make sure robot existed in previous step
                old_carrying = any(task['assigned_to'] == robot_id and task['completed'] 
                                 for task in self.tasks.values())
                if robot['carrying_task'] is None and old_carrying:
                    reward += 100
            
            # Reward for moving towards assigned task
            if robot['assigned_task'] is not None:
                task = self.tasks[robot['assigned_task']]
                target = task['pickup'] if robot['carrying_task'] is None else task['dropoff']
                distance_to_target = self._manhattan_distance(robot['position'], target)
                reward += max(0, 15 - distance_to_target)
            
            # Reward for picking up a task
            if (robot['carrying_task'] is not None and 
                robot_id in old_positions and 
                robot['position'] == old_positions[robot_id]):  # Just picked up
                reward += 50
            
            # Small penalty for each step to encourage efficiency
            reward -= 1
            
            # Penalty for being idle
            reward -= robot['idle_time'] * 0.5
            
            # Penalty for collisions (shared penalty)
            if self.collision_count > 0:
                reward -= 5
            
            rewards[robot_id] = reward
        
        return rewards
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _check_done(self):
        """Check if episode should end"""
        all_tasks_completed = all(task['completed'] for task in self.tasks.values())
        max_steps_reached = self.step_count >= self.max_steps
        return all_tasks_completed or max_steps_reached
    
    def _get_positions_snapshot(self):
        """Get current positions for history tracking"""
        return {
            'step': self.step_count,
            'robots': {rid: robot['position'] for rid, robot in self.robots.items()},
            'dynamic_obstacles': self.dynamic_obstacles.copy(),
            'completed_tasks': len(self.completed_tasks)
        }
    
    def get_metrics(self):
        """Get comprehensive evaluation metrics"""
        completed_count = len(self.completed_tasks)
        completion_rate = completed_count / self.num_tasks if self.num_tasks > 0 else 0
        
        avg_path_length = np.mean([robot['total_distance'] for robot in self.robots.values()])
        max_path_length = max([robot['total_distance'] for robot in self.robots.values()])
        avg_idle_time = np.mean([robot['idle_time'] for robot in self.robots.values()])
        
        throughput = completed_count / self.step_count if self.step_count > 0 else 0
        
        # Calculate path efficiency (ratio of straight-line distance to actual path)
        path_efficiencies = []
        for robot in self.robots.values():
            if len(robot['path_history']) > 1:
                straight_line = self._manhattan_distance(robot['path_history'][0], robot['path_history'][-1])
                actual_path = robot['total_distance']
                if actual_path > 0:
                    efficiency = straight_line / actual_path
                    path_efficiencies.append(efficiency)
        
        avg_path_efficiency = np.mean(path_efficiencies) if path_efficiencies else 0
        
        return {
            'task_completion_rate': completion_rate,
            'completed_tasks': completed_count,
            'total_tasks': self.num_tasks,
            'collision_count': self.collision_count,
            'collision_rate': self.collision_count / self.step_count if self.step_count > 0 else 0,
            'avg_path_length': avg_path_length,
            'max_path_length': max_path_length,
            'total_distance': self.total_distance_traveled,
            'avg_idle_time': avg_idle_time,
            'throughput': throughput,
            'avg_path_efficiency': avg_path_efficiency,
            'total_steps': self.step_count,
            'success': completion_rate > 0.8 and self.collision_count < 5
        }
    
    def render(self, save_path=None, show_paths=True, figsize=(12, 10)):
        """Render current state of the environment"""
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        # Draw grid
        for i in range(self.grid_size[0] + 1):
            plt.axhline(y=i-0.5, color='lightgray', linewidth=0.5)
        for j in range(self.grid_size[1] + 1):
            plt.axvline(x=j-0.5, color='lightgray', linewidth=0.5)
        
        # Draw static obstacles
        for obs in self.static_obstacles:
            rect = patches.Rectangle((obs[1]-0.45, obs[0]-0.45), 0.9, 0.9, 
                                   linewidth=1, edgecolor='black', facecolor='gray', alpha=0.8)
            ax.add_patch(rect)
        
        # Draw dynamic obstacles
        for obs in self.dynamic_obstacles.values():
            rect = patches.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8, 
                                   linewidth=2, edgecolor='red', facecolor='orange', alpha=0.7)
            ax.add_patch(rect)
        
        # Draw tasks
        task_colors = ['blue', 'green', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'lime']
        for i, (task_id, task) in enumerate(self.tasks.items()):
            color = task_colors[i % len(task_colors)]
            alpha = 0.3 if task['completed'] else 0.8
            
            # Pickup location (square)
            plt.scatter(task['pickup'][1], task['pickup'][0], s=200, c=color, 
                       marker='s', alpha=alpha, edgecolors='black', linewidth=2)
            # Dropoff location (diamond)
            plt.scatter(task['dropoff'][1], task['dropoff'][0], s=200, c=color, 
                       marker='D', alpha=alpha, edgecolors='black', linewidth=2)
            
            if not task['completed']:
                # Draw line connecting pickup to dropoff
                plt.plot([task['pickup'][1], task['dropoff'][1]], 
                        [task['pickup'][0], task['dropoff'][0]], 
                        '--', color=color, alpha=0.5, linewidth=2)
        
        # Draw robot paths
        robot_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        if show_paths:
            for i, (robot_id, robot) in enumerate(self.robots.items()):
                color = robot_colors[i % len(robot_colors)]
                if len(robot['path_history']) > 1:
                    path_y = [pos[1] for pos in robot['path_history']]
                    path_x = [pos[0] for pos in robot['path_history']]
                    plt.plot(path_y, path_x, '-', color=color, alpha=0.4, linewidth=2)
        
        # Draw robots
        for i, (robot_id, robot) in enumerate(self.robots.items()):
            color = robot_colors[i % len(robot_colors)]
            pos = robot['position']
            
            # Robot circle
            circle = patches.Circle((pos[1], pos[0]), 0.3, 
                                  facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
            ax.add_patch(circle)
            
            # Robot ID
            plt.text(pos[1], pos[0], str(robot_id), ha='center', va='center', 
                    fontweight='bold', color='white', fontsize=12)
            
            # Show what robot is carrying/assigned to
            if robot['carrying_task'] is not None:
                plt.text(pos[1], pos[0]-0.7, f"C:{robot['carrying_task']}", 
                        ha='center', va='center', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.1", facecolor='yellow', alpha=0.7))
            elif robot['assigned_task'] is not None:
                plt.text(pos[1], pos[0]-0.7, f"A:{robot['assigned_task']}", 
                        ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor='lightblue', alpha=0.7))
        
        # Set limits and labels
        plt.xlim(-0.5, self.grid_size[1] - 0.5)
        plt.ylim(-0.5, self.grid_size[0] - 0.5)
        plt.gca().invert_yaxis()
        
        # Title with current metrics
        metrics = self.get_metrics()
        plt.title(f'Multi-Robot Environment - Step {self.step_count}\n'
                 f'Tasks: {metrics["completed_tasks"]}/{metrics["total_tasks"]} | '
                 f'Collisions: {metrics["collision_count"]} | '
                 f'Completion: {metrics["task_completion_rate"]:.1%}',
                 fontsize=14, pad=20)
        
        plt.xlabel('Y Coordinate', fontsize=12)
        plt.ylabel('X Coordinate', fontsize=12)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                      markersize=10, label='Task Pickup', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', 
                      markersize=10, label='Task Dropoff', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                      markersize=10, label='Static Obstacle', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
                      markersize=10, label='Dynamic Obstacle', markeredgecolor='red'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=12, label='Robot', markeredgecolor='black')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Environment visualization saved to {save_path}")
        
        plt.show()
    
    def run_interactive_demo(self, manual_control=False):
        """
        Run an interactive demo of the environment
        
        Args:
            manual_control: If True, allows manual control of robots via keyboard
        """
        print("🎮 Starting Interactive Demo")
        print("="*50)
        
        if manual_control:
            print("Manual Control Mode:")
            print("  Robot 0: WASD keys (W=up, A=left, S=down, D=right, Q=stay)")
            print("  Robot 1: Arrow keys (↑=up, ←=left, ↓=down, →=right, Space=stay)")
            print("  Other robots: Random actions")
            print("  Press Enter to step, 'quit' to exit")
        else:
            print("Automatic Demo Mode:")
            print("  All robots take random actions")
            print("  Press Enter to step, 'quit' to exit")
        
        print("="*50)
        
        step = 0
        while not self._check_done() and step < 100:
            print(f"\n--- Step {step} ---")
            
            # Display current environment
            self.render(show_paths=False)
            
            # Get actions
            actions = {}
            if manual_control:
                # Manual control for robot 0 and 1
                print("Enter actions for robots:")
                for robot_id in range(min(2, self.num_robots)):
                    while True:
                        action_input = input(f"Robot {robot_id} action (0-4 or WASD): ").strip().lower()
                        if action_input in ['quit', 'q']:
                            return
                        
                        # Convert input to action
                        action_map = {
                            '0': 0, 'q': 0, ' ': 0,  # Stay
                            '1': 1, 'w': 1, '↑': 1,   # Up
                            '2': 2, 'd': 2, '→': 2,   # Right
                            '3': 3, 's': 3, '↓': 3,   # Down
                            '4': 4, 'a': 4, '←': 4    # Left
                        }
                        
                        if action_input in action_map:
                            actions[robot_id] = action_map[action_input]
                            break
                        else:
                            print("Invalid action! Use 0-4 or WASD")
                
                # Random actions for remaining robots
                for robot_id in range(2, self.num_robots):
                    actions[robot_id] = random.randint(0, 4)
                    
            else:
                # Random actions for all robots
                for robot_id in self.robots.keys():
                    actions[robot_id] = random.randint(0, 4)
                
                # Wait for user input to continue
                user_input = input("Press Enter to continue or 'quit' to exit: ").strip().lower()
                if user_input in ['quit', 'q']:
                    return
            
            # Execute step
            state, rewards, done, info = self.step(actions)
            
            # Print step information
            print(f"Actions taken: {[(rid, self.action_names[action]) for rid, action in actions.items()]}")
            print(f"Rewards: {rewards}")
            print(f"Completed tasks: {info['tasks_completed']}/{info['total_tasks']}")
            
            if info['collisions_this_step']:
                print("⚠️  Collision detected!")
            
            step += 1
            
            if done:
                print(f"\n🏁 Episode finished after {step} steps!")
                final_metrics = self.get_metrics()
                print("Final Results:")
                print(f"  Task completion rate: {final_metrics['task_completion_rate']:.1%}")
                print(f"  Total collisions: {final_metrics['collision_count']}")
                print(f"  Average path length: {final_metrics['avg_path_length']:.2f}")
                print(f"  Success: {final_metrics['success']}")
                break
        
        # Final render
        self.render(show_paths=True)
    
    def benchmark_random_policy(self, episodes=10):
        """
        Benchmark the environment using random policy
        
        Args:
            episodes: Number of episodes to run
            
        Returns:
            dict: Aggregated metrics across episodes
        """
        print(f"🧪 Running benchmark with random policy ({episodes} episodes)")
        print("="*60)
        
        all_metrics = []
        
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            
            # Reset environment
            self.reset()
            
            # Run episode with random actions
            done = False
            step_count = 0
            while not done and step_count < self.max_steps:
                # Random actions for all robots
                actions = {robot_id: random.randint(0, 4) for robot_id in self.robots.keys()}
                
                # Step environment
                state, rewards, done, info = self.step(actions)
                step_count += 1
                
                # Print progress every 50 steps
                if step_count % 50 == 0:
                    print(f"  Step {step_count}: {info['tasks_completed']}/{info['total_tasks']} tasks completed")
            
            # Get episode metrics
            metrics = self.get_metrics()
            all_metrics.append(metrics)
            
            print(f"  Episode {episode + 1} Results:")
            print(f"    Completion rate: {metrics['task_completion_rate']:.1%}")
            print(f"    Collisions: {metrics['collision_count']}")
            print(f"    Steps: {metrics['total_steps']}")
            print(f"    Success: {'✅' if metrics['success'] else '❌'}")
        
        # Aggregate results
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if isinstance(all_metrics[0][key], (int, float)):
                avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in all_metrics])
                avg_metrics[f'std_{key}'] = np.std([m[key] for m in all_metrics])
        
        # Success rate
        success_count = sum(1 for m in all_metrics if m['success'])
        avg_metrics['success_rate'] = success_count / episodes
        
        print(f"\n📊 Benchmark Results (Random Policy, {episodes} episodes)")
        print("="*60)
        print(f"Success Rate: {avg_metrics['success_rate']:.1%}")
        print(f"Avg Task Completion: {avg_metrics['avg_task_completion_rate']:.1%} ± {avg_metrics['std_task_completion_rate']:.1%}")
        print(f"Avg Collisions: {avg_metrics['avg_collision_count']:.1f} ± {avg_metrics['std_collision_count']:.1f}")
        print(f"Avg Steps: {avg_metrics['avg_total_steps']:.1f} ± {avg_metrics['std_total_steps']:.1f}")
        print(f"Avg Path Efficiency: {avg_metrics['avg_avg_path_efficiency']:.3f} ± {avg_metrics['std_avg_path_efficiency']:.3f}")
        
        return avg_metrics, all_metrics
    
    def save_configuration(self, filepath):
        """Save current environment configuration to JSON file"""
        config = {
            'grid_size': self.grid_size,
            'num_robots': self.num_robots,
            'num_tasks': self.num_tasks,
            'max_steps': self.max_steps,
            'robots': {k: v.copy() for k, v in self.robots.items()},
            'tasks': {k: v.copy() for k, v in self.tasks.items()},
            'static_obstacles': list(self.static_obstacles),
            'dynamic_obstacles': {k: v for k, v in self.dynamic_obstacles.items()}
        }
        
        # Convert tuples to lists for JSON serialization
        for robot in config['robots'].values():
            robot['position'] = list(robot['position'])
            robot['path_history'] = [list(pos) for pos in robot['path_history']]
        
        for task in config['tasks'].values():
            task['pickup'] = list(task['pickup'])
            task['dropoff'] = list(task['dropoff'])
        
        config['static_obstacles'] = [list(pos) for pos in config['static_obstacles']]
        config['dynamic_obstacles'] = {k: list(v) for k, v in config['dynamic_obstacles'].items()}
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Environment configuration saved to {filepath}")
    
    def load_configuration(self, filepath):
        """Load environment configuration from JSON file"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Restore environment parameters
        self.grid_size = tuple(config['grid_size'])
        self.num_robots = config['num_robots']
        self.num_tasks = config['num_tasks']
        self.max_steps = config['max_steps']
        
        # Restore state
        self.robots = {}
        for k, v in config['robots'].items():
            self.robots[int(k)] = {
                'position': tuple(v['position']),
                'assigned_task': v['assigned_task'],
                'carrying_task': v['carrying_task'],
                'idle_time': v['idle_time'],
                'total_distance': v['total_distance'],
                'path_history': [tuple(pos) for pos in v['path_history']]
            }
        
        self.tasks = {}
        for k, v in config['tasks'].items():
            self.tasks[int(k)] = {
                'pickup': tuple(v['pickup']),
                'dropoff': tuple(v['dropoff']),
                'completed': v['completed'],
                'assigned_to': v['assigned_to'],
                'priority': v['priority']
            }
        
        self.static_obstacles = set(tuple(pos) for pos in config['static_obstacles'])
        self.dynamic_obstacles = {int(k): tuple(v) for k, v in config['dynamic_obstacles'].items()}
        
        # Reset counters
        self.step_count = 0
        self.collision_count = 0
        self.total_distance_traveled = 0
        self.completed_tasks = set()
        self.position_history = []
        self.collision_history = []
        
        print(f"Environment configuration loaded from {filepath}")
    
    def create_animation(self, save_path="robot_animation.gif", interval=500, max_frames=100):
        """
        Create an animated visualization of robot movements
        
        Args:
            save_path: Path to save the animation
            interval: Time between frames in milliseconds
            max_frames: Maximum number of frames to include
        """
        if len(self.position_history) < 2:
            print("Not enough position history to create animation. Run some steps first!")
            return
        
        print(f"🎬 Creating animation with {min(len(self.position_history), max_frames)} frames...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Limit frames for performance
        frames = min(len(self.position_history), max_frames)
        history_subset = self.position_history[:frames]
        
        def animate_frame(frame_idx):
            ax.clear()
            
            # Get current frame data
            frame_data = history_subset[frame_idx]
            
            # Draw grid
            for i in range(self.grid_size[0] + 1):
                ax.axhline(y=i-0.5, color='lightgray', linewidth=0.5)
            for j in range(self.grid_size[1] + 1):
                ax.axvline(x=j-0.5, color='lightgray', linewidth=0.5)
            
            # Draw static obstacles
            for obs in self.static_obstacles:
                rect = patches.Rectangle((obs[1]-0.45, obs[0]-0.45), 0.9, 0.9, 
                                       linewidth=1, edgecolor='black', facecolor='gray', alpha=0.8)
                ax.add_patch(rect)
            
            # Draw dynamic obstacles for this frame
            for obs in frame_data['dynamic_obstacles'].values():
                rect = patches.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8, 
                                       linewidth=2, edgecolor='red', facecolor='orange', alpha=0.7)
                ax.add_patch(rect)
            
            # Draw tasks
            task_colors = ['blue', 'green', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'lime']
            completed_tasks_at_frame = frame_data['completed_tasks']
            
            for i, (task_id, task) in enumerate(self.tasks.items()):
                color = task_colors[i % len(task_colors)]
                alpha = 0.3 if task_id in self.completed_tasks else 0.8
                
                # Pickup location (square)
                ax.scatter(task['pickup'][1], task['pickup'][0], s=200, c=color, 
                          marker='s', alpha=alpha, edgecolors='black', linewidth=2)
                # Dropoff location (diamond)
                ax.scatter(task['dropoff'][1], task['dropoff'][0], s=200, c=color, 
                          marker='D', alpha=alpha, edgecolors='black', linewidth=2)
            
            # Draw robot trails (fade over time)
            robot_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
            trail_length = 10
            
            for i, robot_id in enumerate(frame_data['robots'].keys()):
                color = robot_colors[i % len(robot_colors)]
                
                # Draw trail
                trail_positions = []
                for prev_frame_idx in range(max(0, frame_idx - trail_length), frame_idx):
                    if robot_id in history_subset[prev_frame_idx]['robots']:
                        trail_positions.append(history_subset[prev_frame_idx]['robots'][robot_id])
                
                if len(trail_positions) > 1:
                    trail_y = [pos[1] for pos in trail_positions]
                    trail_x = [pos[0] for pos in trail_positions]
                    ax.plot(trail_y, trail_x, '-', color=color, alpha=0.3, linewidth=2)
            
            # Draw robots at current positions
            for i, (robot_id, pos) in enumerate(frame_data['robots'].items()):
                color = robot_colors[i % len(robot_colors)]
                
                # Robot circle
                circle = patches.Circle((pos[1], pos[0]), 0.3, 
                                      facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
                ax.add_patch(circle)
                
                # Robot ID
                ax.text(pos[1], pos[0], str(robot_id), ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=12)
            
            # Set limits and labels
            ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
            ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
            ax.invert_yaxis()
            
            # Title
            ax.set_title(f'Multi-Robot Environment - Step {frame_data["step"]}\n'
                        f'Completed Tasks: {completed_tasks_at_frame}', 
                        fontsize=14, pad=20)
            
            ax.set_xlabel('Y Coordinate', fontsize=12)
            ax.set_ylabel('X Coordinate', fontsize=12)
        
        # Create animation
        anim = FuncAnimation(fig, animate_frame, frames=frames, interval=interval, repeat=True)
        
        # Save animation
        anim.save(save_path, writer='pillow')
        print(f"Animation saved to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Multi-Robot Environment Framework")
    print("="*50)
    
    # Create environment
    env = MultiRobotEnvironment(grid_size=(12, 12), num_robots=3, num_tasks=4, seed=42)
    
    # Show initial state
    print("\n📸 Initial Environment State:")
    env.render(show_paths=False)
    
    # Run a few random steps
    print("\n🎲 Running 10 random steps...")
    for step in range(10):
        # Random actions
        actions = {robot_id: random.randint(0, 4) for robot_id in env.robots.keys()}
        state, rewards, done, info = env.step(actions)
        
        print(f"Step {step + 1}: Actions {actions}, Rewards {rewards}")
        if done:
            print("Episode completed!")
            break
    
    # Show final state
    print("\n📸 Final Environment State:")
    env.render(show_paths=True)
    
    # Show metrics
    print("\n📊 Environment Metrics:")
    metrics = env.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Demonstrate interactive demo (uncomment to try)
    # env.run_interactive_demo(manual_control=False)
    
    # Demonstrate benchmark (uncomment to try)
    # env.benchmark_random_policy(episodes=5)
    
    print("\n✅ Framework demonstration complete!")
    print("💡 This environment is ready to be used with any path planning algorithm!")