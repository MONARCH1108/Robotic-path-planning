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
    Grid-based environment for multi-robot path planning
    """
    def __init__(self, grid_size=(15, 15), num_robots=3, num_tasks=5):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.max_steps = 200  # Maximum steps per episode
        
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
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.step_count = 0
        self.collision_count = 0
        self.total_distance_traveled = 0
        self.completed_tasks = set()
        
        # Generate static obstacles
        self._generate_static_obstacles()
        
        # Initialize robots
        self.robots = {}
        for i in range(self.num_robots):
            while True:
                pos = (random.randint(0, self.grid_size[0]-1), 
                       random.randint(0, self.grid_size[1]-1))
                if pos not in self.static_obstacles and pos not in [r['position'] for r in self.robots.values()]:
                    self.robots[i] = {
                        'position': pos,
                        'assigned_task': None,
                        'carrying_task': None,
                        'idle_time': 0,
                        'total_distance': 0
                    }
                    break
        
        # Generate tasks (pick-up and drop-off locations)
        self.tasks = {}
        for i in range(self.num_tasks):
            while True:
                pickup = (random.randint(0, self.grid_size[0]-1), 
                         random.randint(0, self.grid_size[1]-1))
                dropoff = (random.randint(0, self.grid_size[0]-1), 
                          random.randint(0, self.grid_size[1]-1))
                
                occupied_positions = (set([r['position'] for r in self.robots.values()]) | 
                                    self.static_obstacles |
                                    set([t['pickup'] for t in self.tasks.values()]) |
                                    set([t['dropoff'] for t in self.tasks.values()]))
                
                if (pickup not in occupied_positions and dropoff not in occupied_positions 
                    and pickup != dropoff):
                    self.tasks[i] = {
                        'pickup': pickup,
                        'dropoff': dropoff,
                        'completed': False,
                        'assigned_to': None
                    }
                    break
        
        # Generate dynamic obstacles
        self._update_dynamic_obstacles()
        
        return self.get_state()
    
    def _generate_static_obstacles(self):
        """Generate random static obstacles"""
        self.static_obstacles = set()
        num_obstacles = random.randint(5, 15)
        
        for _ in range(num_obstacles):
            while True:
                pos = (random.randint(0, self.grid_size[0]-1), 
                       random.randint(0, self.grid_size[1]-1))
                if pos not in self.static_obstacles:
                    self.static_obstacles.add(pos)
                    break
    
    def _update_dynamic_obstacles(self):
        """Update dynamic obstacles (moving machinery simulation)"""
        self.dynamic_obstacles = {}
        num_dynamic = random.randint(1, 3)
        
        for i in range(num_dynamic):
            pos = (random.randint(0, self.grid_size[0]-1), 
                   random.randint(0, self.grid_size[1]-1))
            self.dynamic_obstacles[i] = pos
    
    def get_state(self):
        """Get current state representation"""
        state = {
            'robots': self.robots.copy(),
            'tasks': self.tasks.copy(),
            'static_obstacles': self.static_obstacles.copy(),
            'dynamic_obstacles': self.dynamic_obstacles.copy(),
            'step_count': self.step_count,
            'completed_tasks': self.completed_tasks.copy()
        }
        return state
    
    def step(self, actions_dict):
        """Execute actions for all robots"""
        self.step_count += 1
        
        # Store old positions for collision detection
        old_positions = {robot_id: robot['position'] for robot_id, robot in self.robots.items()}
        
        # Execute actions
        for robot_id, action in actions_dict.items():
            if robot_id in self.robots:
                self._execute_robot_action(robot_id, action)
        
        # Check for collisions
        self._check_collisions(old_positions)
        
        # Update dynamic obstacles
        if self.step_count % 10 == 0:  # Update every 10 steps
            self._update_dynamic_obstacles()
        
        # Handle task assignments and completions
        self._handle_tasks()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Check if episode is done
        done = self._check_done()
        
        return self.get_state(), rewards, done
    
    def _execute_robot_action(self, robot_id, action):
        """Execute action for a specific robot"""
        robot = self.robots[robot_id]
        old_pos = robot['position']
        
        # Get new position
        dx, dy = self.actions[action]
        new_x = max(0, min(self.grid_size[0]-1, old_pos[0] + dx))
        new_y = max(0, min(self.grid_size[1]-1, old_pos[1] + dy))
        new_pos = (new_x, new_y)
        
        # Check if new position is valid (not an obstacle)
        if (new_pos not in self.static_obstacles and 
            new_pos not in self.dynamic_obstacles.values()):
            robot['position'] = new_pos
            
            # Calculate distance traveled
            if new_pos != old_pos:
                distance = np.sqrt((new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2)
                robot['total_distance'] += distance
                self.total_distance_traveled += distance
        
        # Update idle time
        if robot['position'] == old_pos:
            robot['idle_time'] += 1
    
    def _check_collisions(self, old_positions):
        """Check for robot-robot collisions"""
        positions = [robot['position'] for robot in self.robots.values()]
        
        # Check for multiple robots at same position
        if len(positions) != len(set(positions)):
            self.collision_count += 1
        
        # Check for robots swapping positions (more sophisticated collision)
        robot_ids = list(self.robots.keys())
        for i in range(len(robot_ids)):
            for j in range(i+1, len(robot_ids)):
                robot1_id, robot2_id = robot_ids[i], robot_ids[j]
                old_pos1, old_pos2 = old_positions[robot1_id], old_positions[robot2_id]
                new_pos1, new_pos2 = self.robots[robot1_id]['position'], self.robots[robot2_id]['position']
                
                if old_pos1 == new_pos2 and old_pos2 == new_pos1:
                    self.collision_count += 1
    
    def _handle_tasks(self):
        """Handle task assignments and completions"""
        # Assign tasks to robots
        for robot_id, robot in self.robots.items():
            if robot['assigned_task'] is None and robot['carrying_task'] is None:
                # Find nearest unassigned task
                available_tasks = [tid for tid, task in self.tasks.items() 
                                 if not task['completed'] and task['assigned_to'] is None]
                
                if available_tasks:
                    robot_pos = robot['position']
                    nearest_task = min(available_tasks, 
                                     key=lambda tid: self._manhattan_distance(robot_pos, self.tasks[tid]['pickup']))
                    
                    robot['assigned_task'] = nearest_task
                    self.tasks[nearest_task]['assigned_to'] = robot_id
            
            # Check if robot reached pickup location
            if (robot['assigned_task'] is not None and robot['carrying_task'] is None):
                task = self.tasks[robot['assigned_task']]
                if robot['position'] == task['pickup']:
                    robot['carrying_task'] = robot['assigned_task']
                    robot['assigned_task'] = None
            
            # Check if robot reached dropoff location
            if robot['carrying_task'] is not None:
                task = self.tasks[robot['carrying_task']]
                if robot['position'] == task['dropoff']:
                    task['completed'] = True
                    self.completed_tasks.add(robot['carrying_task'])
                    robot['carrying_task'] = None
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _calculate_rewards(self):
        """Calculate rewards for each robot"""
        rewards = {}
        
        for robot_id, robot in self.robots.items():
            reward = 0
            
            # Reward for task completion
            if robot['carrying_task'] is not None:
                task = self.tasks[robot['carrying_task']]
                if task['completed']:
                    reward += 100  # Large reward for completing task
            
            # Reward for moving towards assigned task
            if robot['assigned_task'] is not None:
                task = self.tasks[robot['assigned_task']]
                target = task['pickup'] if robot['carrying_task'] is None else task['dropoff']
                distance_to_target = self._manhattan_distance(robot['position'], target)
                reward += max(0, 20 - distance_to_target)  # Closer to target = higher reward
            
            # Penalty for collisions
            reward -= self.collision_count * 10
            
            # Small penalty for each step to encourage efficiency
            reward -= 1
            
            # Penalty for being idle
            if robot['idle_time'] > 0:
                reward -= robot['idle_time'] * 0.5
            
            rewards[robot_id] = reward
        
        return rewards
    
    def _check_done(self):
        """Check if episode is complete"""
        all_tasks_completed = all(task['completed'] for task in self.tasks.values())
        max_steps_reached = self.step_count >= self.max_steps
        return all_tasks_completed or max_steps_reached
    
    def get_metrics(self):
        """Get evaluation metrics"""
        completed_count = len(self.completed_tasks)
        completion_rate = completed_count / self.num_tasks if self.num_tasks > 0 else 0
        
        avg_path_length = np.mean([robot['total_distance'] for robot in self.robots.values()])
        avg_idle_time = np.mean([robot['idle_time'] for robot in self.robots.values()])
        
        throughput = completed_count / self.step_count if self.step_count > 0 else 0
        
        return {
            'avg_path_length': avg_path_length,
            'total_distance': self.total_distance_traveled,
            'collision_count': self.collision_count,
            'task_completion_rate': completion_rate,
            'completed_tasks': completed_count,
            'total_tasks': self.num_tasks,
            'throughput': throughput,
            'avg_idle_time': avg_idle_time,
            'total_steps': self.step_count
        }


class QLearningAgent:
    """
    Q-Learning agent for multi-robot path planning
    """
    def __init__(self, robot_id, grid_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.robot_id = robot_id
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(5))  # 5 actions
        self.state_visits = defaultdict(int)
        
    def get_state_key(self, state):
        """Convert state to a hashable key for Q-table"""
        robot = state['robots'][self.robot_id]
        robot_pos = robot['position']
        
        # Include information about assigned task
        target_pos = None
        if robot['assigned_task'] is not None:
            task = state['tasks'][robot['assigned_task']]
            target_pos = task['pickup'] if robot['carrying_task'] is None else task['dropoff']
        elif robot['carrying_task'] is not None:
            task = state['tasks'][robot['carrying_task']]
            target_pos = task['dropoff']
        
        # Include nearby obstacles and other robots
        nearby_obstacles = []
        nearby_robots = []
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                check_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
                if (0 <= check_pos[0] < self.grid_size[0] and 
                    0 <= check_pos[1] < self.grid_size[1]):
                    
                    if (check_pos in state['static_obstacles'] or 
                        check_pos in state['dynamic_obstacles'].values()):
                        nearby_obstacles.append((dx, dy))
                    
                    for other_id, other_robot in state['robots'].items():
                        if other_id != self.robot_id and other_robot['position'] == check_pos:
                            nearby_robots.append((dx, dy))
        
        state_key = (
            robot_pos,
            target_pos,
            tuple(sorted(nearby_obstacles)),
            tuple(sorted(nearby_robots)),
            robot['carrying_task'] is not None
        )
        
        return state_key
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        self.state_visits[state_key] += 1
        
        if random.random() < self.epsilon:
            return random.randint(0, 4)  # Random action
        else:
            return np.argmax(self.q_table[state_key])  # Greedy action
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_q_table_size(self):
        """Get the number of states in Q-table"""
        return len(self.q_table)


class MultiRobotQLearning:
    """
    Multi-robot coordination using Q-Learning
    """
    def __init__(self, env, learning_params=None):
        self.env = env
        self.agents = {}
        
        # Default learning parameters
        if learning_params is None:
            learning_params = {
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01
            }
        
        # Initialize Q-learning agents
        for robot_id in range(env.num_robots):
            self.agents[robot_id] = QLearningAgent(
                robot_id=robot_id,
                grid_size=env.grid_size,
                **learning_params
            )
        
        # Training history
        self.training_history = []
    
    def train(self, episodes=1000, verbose=True):
        """Train the multi-robot system using Q-learning"""
        episode_metrics = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = {robot_id: 0 for robot_id in self.agents.keys()}
            
            while True:
                # Get actions from all agents
                actions = {}
                for robot_id, agent in self.agents.items():
                    actions[robot_id] = agent.choose_action(state)
                
                # Execute actions
                next_state, rewards, done = self.env.step(actions)
                
                # Update Q-values for all agents
                for robot_id, agent in self.agents.items():
                    agent.update_q_value(state, actions[robot_id], 
                                       rewards[robot_id], next_state)
                    episode_reward[robot_id] += rewards[robot_id]
                
                state = next_state
                
                if done:
                    break
            
            # Store episode metrics
            metrics = self.env.get_metrics()
            metrics['episode'] = episode
            metrics['total_reward'] = sum(episode_reward.values())
            metrics['avg_epsilon'] = np.mean([agent.epsilon for agent in self.agents.values()])
            episode_metrics.append(metrics)
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}")
                print(f"  Completion Rate: {metrics['task_completion_rate']:.2%}")
                print(f"  Collisions: {metrics['collision_count']}")
                print(f"  Avg Path Length: {metrics['avg_path_length']:.2f}")
                print(f"  Total Reward: {metrics['total_reward']:.2f}")
                print(f"  Avg Epsilon: {metrics['avg_epsilon']:.3f}")
                print()
        
        self.training_history = episode_metrics
        return episode_metrics
    
    def test(self, episodes=10):
        """Test the trained agents"""
        # Set epsilon to 0 for testing (no exploration)
        original_epsilons = {}
        for robot_id, agent in self.agents.items():
            original_epsilons[robot_id] = agent.epsilon
            agent.epsilon = 0.0
        
        test_metrics = []
        
        for episode in range(episodes):
            state = self.env.reset()
            
            while True:
                actions = {}
                for robot_id, agent in self.agents.items():
                    actions[robot_id] = agent.choose_action(state)
                
                state, _, done = self.env.step(actions)
                
                if done:
                    break
            
            metrics = self.env.get_metrics()
            metrics['episode'] = episode
            test_metrics.append(metrics)
        
        # Restore original epsilons
        for robot_id, agent in self.agents.items():
            agent.epsilon = original_epsilons[robot_id]
        
        return test_metrics
    
    def get_agent_info(self):
        """Get information about trained agents"""
        info = {}
        for robot_id, agent in self.agents.items():
            info[robot_id] = {
                'q_table_size': agent.get_q_table_size(),
                'epsilon': agent.epsilon,
                'total_state_visits': sum(agent.state_visits.values())
            }
        return info


class Visualizer:
    """
    Visualization for multi-robot environment
    """
    def __init__(self, env):
        self.env = env
        self.fig = None
        self.ax = None
    
    def plot_environment(self, save_path=None):
        """Plot current state of environment"""
        plt.figure(figsize=(12, 10))
        
        # Create grid
        for i in range(self.env.grid_size[0] + 1):
            plt.axvline(x=i-0.5, color='lightgray', linewidth=0.5)
        for j in range(self.env.grid_size[1] + 1):
            plt.axhline(y=j-0.5, color='lightgray', linewidth=0.5)
        
        # Plot static obstacles
        for obs in self.env.static_obstacles:
            rect = patches.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8, 
                                   linewidth=1, edgecolor='black', facecolor='black')
            plt.gca().add_patch(rect)
        
        # Plot dynamic obstacles
        for obs in self.env.dynamic_obstacles.values():
            rect = patches.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8, 
                                   linewidth=1, edgecolor='red', facecolor='red', alpha=0.7)
            plt.gca().add_patch(rect)
        
        # Plot tasks
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, (task_id, task) in enumerate(self.env.tasks.items()):
            color = colors[i % len(colors)]
            
            if not task['completed']:
                # Pickup location
                plt.plot(task['pickup'][1], task['pickup'][0], 's', 
                        color=color, markersize=10, alpha=0.7)
                # Dropoff location
                plt.plot(task['dropoff'][1], task['dropoff'][0], 'D', 
                        color=color, markersize=10, alpha=0.7)
                # Connect pickup to dropoff
                plt.plot([task['pickup'][1], task['dropoff'][1]], 
                        [task['pickup'][0], task['dropoff'][0]], 
                        '--', color=color, alpha=0.5)
        
        # Plot robots
        robot_colors = ['red', 'blue', 'green', 'yellow', 'purple']
        for i, (robot_id, robot) in enumerate(self.env.robots.items()):
            color = robot_colors[i % len(robot_colors)]
            pos = robot['position']
            
            plt.plot(pos[1], pos[0], 'o', color=color, markersize=15, 
                    markeredgecolor='black', markeredgewidth=2)
            plt.text(pos[1], pos[0], str(robot_id), ha='center', va='center', 
                    fontweight='bold', color='white')
        
        plt.xlim(-0.5, self.env.grid_size[1] - 0.5)
        plt.ylim(-0.5, self.env.grid_size[0] - 0.5)
        plt.gca().invert_yaxis()
        plt.title(f'Multi-Robot Environment - Step {self.env.step_count}')
        plt.xlabel('Y Coordinate')
        plt.ylabel('X Coordinate')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                      markersize=8, label='Pickup Location'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', 
                      markersize=8, label='Dropoff Location'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', 
                      markersize=8, label='Static Obstacle'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=8, label='Dynamic Obstacle'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Robot')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_metrics(self, training_history, save_path=None):
        """Plot training progress metrics"""
        episodes = [m['episode'] for m in training_history]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Q-Learning Training Progress', fontsize=16)
        
        # Task completion rate
        completion_rates = [m['task_completion_rate'] for m in training_history]
        axes[0, 0].plot(episodes, completion_rates)
        axes[0, 0].set_title('Task Completion Rate')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Completion Rate')
        axes[0, 0].grid(True)
        
        # Collision count
        collisions = [m['collision_count'] for m in training_history]
        axes[0, 1].plot(episodes, collisions)
        axes[0, 1].set_title('Collision Count')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Collisions')
        axes[0, 1].grid(True)
        
        # Average path length
        path_lengths = [m['avg_path_length'] for m in training_history]
        axes[0, 2].plot(episodes, path_lengths)
        axes[0, 2].set_title('Average Path Length')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Path Length')
        axes[0, 2].grid(True)
        
        # Total reward
        rewards = [m['total_reward'] for m in training_history]
        axes[1, 0].plot(episodes, rewards)
        axes[1, 0].set_title('Total Episode Reward')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].grid(True)
        
        # Throughput
        throughputs = [m['throughput'] for m in training_history]
        axes[1, 1].plot(episodes, throughputs)
        axes[1, 1].set_title('System Throughput')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Tasks/Step')
        axes[1, 1].grid(True)
        
        # Epsilon decay
        epsilons = [m['avg_epsilon'] for m in training_history]
        axes[1, 2].plot(episodes, epsilons)
        axes[1, 2].set_title('Average Epsilon (Exploration)')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Epsilon')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def run_qlearning_experiment():
    """
    Complete Q-Learning experiment for multi-robot path planning
    """
    print("=" * 60)
    print("Multi-Robot Q-Learning Path Planning Experiment")
    print("=" * 60)
    
    # Environment parameters
    env_params = {
        'grid_size': (12, 12),
        'num_robots': 3,
        'num_tasks': 6
    }
    
    # Q-Learning parameters
    learning_params = {
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.05
    }
    
    print("Environment Setup:")
    print(f"  Grid Size: {env_params['grid_size']}")
    print(f"  Number of Robots: {env_params['num_robots']}")
    print(f"  Number of Tasks: {env_params['num_tasks']}")
    print()
    
    print("Q-Learning Parameters:")
    for key, value in learning_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Create environment and Q-learning system
    env = MultiRobotEnvironment(**env_params)
    qlearning_system = MultiRobotQLearning(env, learning_params)
    
    # Train the system
    print("Starting Training...")
    training_history = qlearning_system.train(episodes=1000, verbose=True)
    
    # Test the trained system
    print("Testing Trained System...")
    test_results = qlearning_system.test(episodes=20)
    
    # Calculate average test performance
    avg_completion_rate = np.mean([m['task_completion_rate'] for m in test_results])
    avg_collisions = np.mean([m['collision_count'] for m in test_results])
    avg_path_length = np.mean([m['avg_path_length'] for m in test_results])
    avg_throughput = np.mean([m['throughput'] for m in test_results])
    
    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS (Average over 20 episodes)")
    print("=" * 50)
    print(f"Task Completion Rate: {avg_completion_rate:.2%}")
    print(f"Average Collisions: {avg_collisions:.2f}")
    print(f"Average Path Length: {avg_path_length:.2f}")
    print(f"System Throughput: {avg_throughput:.4f} tasks/step")
    
    # Agent information
    agent_info = qlearning_system.get_agent_info()
    print(f"\nAgent Learning Statistics:")
    for robot_id, info in agent_info.items():
        print(f"  Robot {robot_id}:")
        print(f"    Q-table size: {info['q_table_size']} states")
        print(f"    Final epsilon: {info['epsilon']:.3f}")
        print(f"    State visits: {info['total_state_visits']}")
    
    # Create visualizations
    visualizer = Visualizer(env)
    
    # Plot training progress
    print("\nGenerating training progress plots...")
    visualizer.plot_training_metrics(training_history)
    
    # Reset environment and show final state
    print("\nShowing sample environment state...")
    env.reset()
    visualizer.plot_environment()
    
    # Save results to JSON for further analysis
    results = {
        'environment_params': env_params,
        'learning_params': learning_params,
        'training_history': training_history,
        'test_results': test_results,
        'average_performance': {
            'completion_rate': avg_completion_rate,
            'collisions': avg_collisions,
            'path_length': avg_path_length,
            'throughput': avg_throughput
        },
        'agent_info': agent_info
    }
    
    with open('qlearning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'qlearning_results.json'")
    
    return results


def compare_qlearning_parameters():
    """
    Compare different Q-Learning parameter configurations
    """
    print("=" * 60)
    print("Q-Learning Parameter Comparison Experiment")
    print("=" * 60)
    
    # Base environment
    env_params = {
        'grid_size': (10, 10),
        'num_robots': 3,
        'num_tasks': 5
    }
    
    # Different parameter configurations to test
    param_configs = {
        'High Learning Rate': {
            'learning_rate': 0.3,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.05
        },
        'Low Learning Rate': {
            'learning_rate': 0.05,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.05
        },
        'High Discount': {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.05
        },
        'Low Discount': {
            'learning_rate': 0.1,
            'discount_factor': 0.8,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.05
        },
        'Fast Epsilon Decay': {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_decay': 0.99,
            'epsilon_min': 0.01
        },
        'Slow Epsilon Decay': {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.1
        }
    }
    
    comparison_results = {}
    
    for config_name, params in param_configs.items():
        print(f"\nTesting configuration: {config_name}")
        print(f"Parameters: {params}")
        
        # Create fresh environment and system
        env = MultiRobotEnvironment(**env_params)
        qlearning_system = MultiRobotQLearning(env, params)
        
        # Train with fewer episodes for comparison
        training_history = qlearning_system.train(episodes=500, verbose=False)
        
        # Test
        test_results = qlearning_system.test(episodes=10)
        
        # Calculate metrics
        avg_completion_rate = np.mean([m['task_completion_rate'] for m in test_results])
        avg_collisions = np.mean([m['collision_count'] for m in test_results])
        avg_path_length = np.mean([m['avg_path_length'] for m in test_results])
        
        comparison_results[config_name] = {
            'completion_rate': avg_completion_rate,
            'collisions': avg_collisions,
            'path_length': avg_path_length,
            'training_history': training_history,
            'test_results': test_results
        }
        
        print(f"  Completion Rate: {avg_completion_rate:.2%}")
        print(f"  Avg Collisions: {avg_collisions:.2f}")
        print(f"  Avg Path Length: {avg_path_length:.2f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Q-Learning Parameter Configuration Comparison', fontsize=16)
    
    configs = list(comparison_results.keys())
    completion_rates = [comparison_results[config]['completion_rate'] for config in configs]
    collisions = [comparison_results[config]['collisions'] for config in configs]
    path_lengths = [comparison_results[config]['path_length'] for config in configs]
    
    # Completion rates
    axes[0].bar(range(len(configs)), completion_rates)
    axes[0].set_title('Task Completion Rate')
    axes[0].set_ylabel('Completion Rate')
    axes[0].set_xticks(range(len(configs)))
    axes[0].set_xticklabels(configs, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Collisions
    axes[1].bar(range(len(configs)), collisions)
    axes[1].set_title('Average Collisions')
    axes[1].set_ylabel('Collisions')
    axes[1].set_xticks(range(len(configs)))
    axes[1].set_xticklabels(configs, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # Path lengths
    axes[2].bar(range(len(configs)), path_lengths)
    axes[2].set_title('Average Path Length')
    axes[2].set_ylabel('Path Length')
    axes[2].set_xticks(range(len(configs)))
    axes[2].set_xticklabels(configs, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 50)
    print("PARAMETER COMPARISON SUMMARY")
    print("=" * 50)
    
    # Find best configuration for each metric
    best_completion = max(configs, key=lambda x: comparison_results[x]['completion_rate'])
    best_collisions = min(configs, key=lambda x: comparison_results[x]['collisions'])
    best_path = min(configs, key=lambda x: comparison_results[x]['path_length'])
    
    print(f"Best Completion Rate: {best_completion} ({comparison_results[best_completion]['completion_rate']:.2%})")
    print(f"Fewest Collisions: {best_collisions} ({comparison_results[best_collisions]['collisions']:.2f})")
    print(f"Shortest Paths: {best_path} ({comparison_results[best_path]['path_length']:.2f})")
    
    return comparison_results


def analyze_scalability():
    """
    Analyze how Q-Learning performance scales with environment complexity
    """
    print("=" * 60)
    print("Q-Learning Scalability Analysis")
    print("=" * 60)
    
    # Test different environment complexities
    complexity_configs = [
        {'grid_size': (8, 8), 'num_robots': 2, 'num_tasks': 3},
        {'grid_size': (10, 10), 'num_robots': 3, 'num_tasks': 5},
        {'grid_size': (12, 12), 'num_robots': 4, 'num_tasks': 7},
        {'grid_size': (15, 15), 'num_robots': 5, 'num_tasks': 10}
    ]
    
    # Fixed learning parameters
    learning_params = {
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.05
    }
    
    scalability_results = []
    
    for i, env_config in enumerate(complexity_configs):
        complexity_level = f"Level {i+1}"
        print(f"\nTesting {complexity_level}: {env_config}")
        
        env = MultiRobotEnvironment(**env_config)
        qlearning_system = MultiRobotQLearning(env, learning_params)
        
        # Measure training time
        start_time = time.time()
        training_history = qlearning_system.train(episodes=800, verbose=False)
        training_time = time.time() - start_time
        
        # Test performance
        test_results = qlearning_system.test(episodes=10)
        
        # Calculate metrics
        avg_completion_rate = np.mean([m['task_completion_rate'] for m in test_results])
        avg_collisions = np.mean([m['collision_count'] for m in test_results])
        avg_path_length = np.mean([m['avg_path_length'] for m in test_results])
        
        # Get Q-table statistics
        agent_info = qlearning_system.get_agent_info()
        total_states = sum(info['q_table_size'] for info in agent_info.values())
        
        result = {
            'complexity_level': complexity_level,
            'config': env_config,
            'completion_rate': avg_completion_rate,
            'collisions': avg_collisions,
            'path_length': avg_path_length,
            'training_time': training_time,
            'total_q_states': total_states
        }
        
        scalability_results.append(result)
        
        print(f"  Completion Rate: {avg_completion_rate:.2%}")
        print(f"  Avg Collisions: {avg_collisions:.2f}")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Total Q-states: {total_states}")
    
    # Plot scalability results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Q-Learning Scalability Analysis', fontsize=16)
    
    levels = [r['complexity_level'] for r in scalability_results]
    
    # Completion rate vs complexity
    completion_rates = [r['completion_rate'] for r in scalability_results]
    axes[0, 0].plot(levels, completion_rates, 'o-')
    axes[0, 0].set_title('Task Completion Rate vs Complexity')
    axes[0, 0].set_ylabel('Completion Rate')
    axes[0, 0].grid(True)
    
    # Training time vs complexity
    training_times = [r['training_time'] for r in scalability_results]
    axes[0, 1].plot(levels, training_times, 'o-', color='red')
    axes[0, 1].set_title('Training Time vs Complexity')
    axes[0, 1].set_ylabel('Training Time (seconds)')
    axes[0, 1].grid(True)
    
    # Q-table size vs complexity
    q_states = [r['total_q_states'] for r in scalability_results]
    axes[1, 0].plot(levels, q_states, 'o-', color='green')
    axes[1, 0].set_title('Total Q-States vs Complexity')
    axes[1, 0].set_ylabel('Number of Q-States')
    axes[1, 0].grid(True)
    
    # Collisions vs complexity
    collisions = [r['collisions'] for r in scalability_results]
    axes[1, 1].plot(levels, collisions, 'o-', color='orange')
    axes[1, 1].set_title('Collisions vs Complexity')
    axes[1, 1].set_ylabel('Average Collisions')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return scalability_results


# Main execution
if __name__ == "__main__":
    # Run the main Q-Learning experiment
    main_results = run_qlearning_experiment()
    
    # Uncomment the following lines to run additional experiments:
    
    # # Compare different parameter configurations
    # print("\n" + "="*80)
    # param_comparison = compare_qlearning_parameters()
    
    # # Analyze scalability
    # print("\n" + "="*80)
    # scalability_analysis = analyze_scalability()
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)