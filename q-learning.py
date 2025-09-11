import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
from typing import Dict, List, Tuple, Optional
import json

# Import the environment (assuming the environment file is saved as 'multi_robot_environment.py')
from Environment.environment import MultiRobotEnvironment

class MultiRobotQLearning:
    """
    Multi-Agent Q-Learning for Multi-Robot Path Planning and Task Allocation
    
    Each robot maintains its own Q-table and learns independently while
    considering the dynamic environment and other robots' positions.
    """
    
    def __init__(self, env: MultiRobotEnvironment, 
                 learning_rate=0.1, 
                 discount_factor=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995):
        """
        Initialize Q-Learning agents for multi-robot system
        
        Args:
            env: MultiRobotEnvironment instance
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        # Number of actions (0=stay, 1=up, 2=right, 3=down, 4=left)
        self.num_actions = 5
        
        # Q-tables for each robot (separate learning)
        self.q_tables = {}
        for robot_id in range(env.num_robots):
            self.q_tables[robot_id] = defaultdict(lambda: np.zeros(self.num_actions))
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'task_completion_rates': [],
            'collision_counts': [],
            'epsilon_values': [],
            'q_table_sizes': []
        }
        
        print(f"Q-Learning initialized for {env.num_robots} robots")
        print(f"Learning rate: {learning_rate}")
        print(f"Discount factor: {discount_factor}")
        print(f"Epsilon: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")
    
    def get_state_representation(self, robot_id: int, state: dict) -> tuple:
        """
        Convert environment state to a hashable state representation for Q-learning
        
        This creates a compact state representation focusing on:
        - Robot's position
        - Assigned task information
        - Local environment (nearby obstacles and robots)
        - Task completion status
        """
        robot = state['robots'][robot_id]
        robot_pos = robot['position']
        
        # Basic position info
        pos_x, pos_y = robot_pos
        
        # Task information
        assigned_task = robot['assigned_task']
        carrying_task = robot['carrying_task']
        
        # Target position (pickup or dropoff)
        target_pos = (-1, -1)  # Default when no task
        if assigned_task is not None:
            task = state['tasks'][assigned_task]
            target_pos = task['pickup'] if carrying_task is None else task['dropoff']
        elif carrying_task is not None:
            task = state['tasks'][carrying_task]
            target_pos = task['dropoff']
        
        # Distance to target
        if target_pos != (-1, -1):
            target_dist = abs(robot_pos[0] - target_pos[0]) + abs(robot_pos[1] - target_pos[1])
        else:
            target_dist = -1
        
        # Local environment (3x3 area around robot)
        local_env = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip robot's own position
                
                check_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
                
                # Check bounds
                if (check_pos[0] < 0 or check_pos[0] >= state['grid_size'][0] or
                    check_pos[1] < 0 or check_pos[1] >= state['grid_size'][1]):
                    local_env.append(1)  # Out of bounds = obstacle
                    continue
                
                # Check for obstacles
                if (check_pos in state['static_obstacles'] or 
                    check_pos in state['dynamic_obstacles'].values()):
                    local_env.append(1)  # Obstacle
                # Check for other robots
                elif any(other_robot['position'] == check_pos 
                        for other_id, other_robot in state['robots'].items() 
                        if other_id != robot_id):
                    local_env.append(2)  # Other robot
                else:
                    local_env.append(0)  # Free space
        
        # Task completion progress
        completed_tasks = len(state['completed_tasks'])
        total_tasks = len(state['tasks'])
        
        # Simplify state to reduce dimensionality
        # Discretize position relative to grid
        rel_pos_x = min(pos_x // 3, 4)  # Divide grid into regions
        rel_pos_y = min(pos_y // 3, 4)
        
        # Discretize target distance
        if target_dist == -1:
            dist_category = 0  # No target
        elif target_dist <= 2:
            dist_category = 1  # Very close
        elif target_dist <= 5:
            dist_category = 2  # Close
        elif target_dist <= 10:
            dist_category = 3  # Medium
        else:
            dist_category = 4  # Far
        
        # Task state
        task_state = 0  # No task
        if carrying_task is not None:
            task_state = 2  # Carrying task
        elif assigned_task is not None:
            task_state = 1  # Assigned task
        
        # Create state tuple (keep it reasonably sized)
        state_tuple = (
            rel_pos_x,
            rel_pos_y,
            task_state,
            dist_category,
            tuple(local_env[:4]),  # Only use 4 neighboring positions to reduce state space
            min(completed_tasks, 5)  # Cap completed tasks to reduce state space
        )
        
        return state_tuple
    
    def choose_action(self, robot_id: int, state_repr: tuple, training=True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, 4)
        else:
            # Exploitation: best known action
            q_values = self.q_tables[robot_id][state_repr]
            return np.argmax(q_values)
    
    def update_q_table(self, robot_id: int, state: tuple, action: int, 
                      reward: float, next_state: tuple, done: bool):
        """Update Q-table using Q-learning update rule"""
        current_q = self.q_tables[robot_id][state][action]
        
        if done:
            target_q = reward
        else:
            next_q_values = self.q_tables[robot_id][next_state]
            target_q = reward + self.discount_factor * np.max(next_q_values)
        
        # Q-learning update
        self.q_tables[robot_id][state][action] += self.learning_rate * (target_q - current_q)
    
    def calculate_shaped_reward(self, robot_id: int, state: dict, action: int, 
                              reward: float, old_state: dict) -> float:
        """
        Calculate shaped reward to guide learning towards objectives:
        1. Minimizing travel time & path length
        2. Collision-free navigation
        3. Task allocation & coordination
        """
        shaped_reward = reward
        
        robot = state['robots'][robot_id]
        old_robot = old_state['robots'][robot_id] if old_state else robot
        
        robot_pos = robot['position']
        old_pos = old_robot['position']
        
        # 1. TRAVEL TIME & PATH LENGTH OPTIMIZATION
        # Reward for moving towards assigned task
        if robot['assigned_task'] is not None:
            task = state['tasks'][robot['assigned_task']]
            target = task['pickup'] if robot['carrying_task'] is None else task['dropoff']
            
            # Distance-based reward
            old_dist = abs(old_pos[0] - target[0]) + abs(old_pos[1] - target[1])
            new_dist = abs(robot_pos[0] - target[0]) + abs(robot_pos[1] - target[1])
            
            if new_dist < old_dist:
                shaped_reward += 2.0  # Reward for getting closer
            elif new_dist > old_dist:
                shaped_reward -= 1.0  # Penalty for moving away
                
            # Bonus for reaching target
            if robot_pos == target:
                shaped_reward += 10.0
        
        # 2. COLLISION-FREE NAVIGATION
        # Penalty for staying idle too long (encourage movement)
        if robot['idle_time'] > 3:
            shaped_reward -= robot['idle_time'] * 0.5
        
        # Reward for avoiding congested areas
        nearby_robots = sum(1 for other_id, other_robot in state['robots'].items()
                           if other_id != robot_id and 
                           abs(other_robot['position'][0] - robot_pos[0]) + 
                           abs(other_robot['position'][1] - robot_pos[1]) <= 2)
        if nearby_robots > 1:
            shaped_reward -= nearby_robots * 0.5  # Penalty for crowding
        
        # 3. TASK ALLOCATION & COORDINATION
        # Reward for task pickup
        if (robot['carrying_task'] is not None and 
            old_robot['carrying_task'] is None):
            shaped_reward += 15.0
        
        # Large reward for task completion
        old_completed = len(old_state['completed_tasks']) if old_state else 0
        new_completed = len(state['completed_tasks'])
        if new_completed > old_completed:
            shaped_reward += 50.0
        
        # Efficiency bonus (reward for making progress)
        if robot['assigned_task'] is not None or robot['carrying_task'] is not None:
            shaped_reward += 1.0  # Small bonus for having purpose
        
        # Penalty for collisions (already in base reward, but emphasize)
        if state['collision_count'] > (old_state['collision_count'] if old_state else 0):
            shaped_reward -= 10.0
        
        return shaped_reward
    
    def train(self, num_episodes=1000, max_steps_per_episode=300, 
              save_interval=100, verbose=True):
        """
        Train Q-learning agents
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            save_interval: Save progress every N episodes
            verbose: Print training progress
        """
        if verbose:
            print(f"Starting Q-Learning training for {num_episodes} episodes...")
            print(f"Max steps per episode: {max_steps_per_episode}")
        
        best_completion_rate = 0.0
        training_start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            episode_rewards = {robot_id: 0 for robot_id in self.env.robots.keys()}
            episode_length = 0
            
            # Get initial state representations
            state_reprs = {}
            for robot_id in self.env.robots.keys():
                state_reprs[robot_id] = self.get_state_representation(robot_id, state)
            
            done = False
            old_state = None
            
            while not done and episode_length < max_steps_per_episode:
                # Choose actions for all robots
                actions = {}
                for robot_id in self.env.robots.keys():
                    actions[robot_id] = self.choose_action(robot_id, state_reprs[robot_id])
                
                # Execute actions
                old_state = state.copy()
                next_state, rewards, done, info = self.env.step(actions)
                
                # Get next state representations
                next_state_reprs = {}
                for robot_id in self.env.robots.keys():
                    next_state_reprs[robot_id] = self.get_state_representation(robot_id, next_state)
                
                # Update Q-tables and calculate shaped rewards
                for robot_id in self.env.robots.keys():
                    # Calculate shaped reward
                    shaped_reward = self.calculate_shaped_reward(
                        robot_id, next_state, actions[robot_id], 
                        rewards[robot_id], old_state
                    )
                    
                    # Update Q-table
                    self.update_q_table(
                        robot_id, 
                        state_reprs[robot_id], 
                        actions[robot_id], 
                        shaped_reward, 
                        next_state_reprs[robot_id], 
                        done
                    )
                    
                    episode_rewards[robot_id] += shaped_reward
                
                # Update state
                state = next_state
                state_reprs = next_state_reprs
                episode_length += 1
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Record statistics
            metrics = self.env.get_metrics()
            self.training_stats['episode_rewards'].append(sum(episode_rewards.values()))
            self.training_stats['episode_lengths'].append(episode_length)
            self.training_stats['task_completion_rates'].append(metrics['task_completion_rate'])
            self.training_stats['collision_counts'].append(metrics['collision_count'])
            self.training_stats['epsilon_values'].append(self.epsilon)
            self.training_stats['q_table_sizes'].append(
                sum(len(q_table) for q_table in self.q_tables.values())
            )
            
            # Track best performance
            if metrics['task_completion_rate'] > best_completion_rate:
                best_completion_rate = metrics['task_completion_rate']
            
            # Print progress
            if verbose and (episode + 1) % save_interval == 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-save_interval:])
                avg_completion = np.mean(self.training_stats['task_completion_rates'][-save_interval:])
                avg_collisions = np.mean(self.training_stats['collision_counts'][-save_interval:])
                
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Completion Rate: {avg_completion:.1%}")
                print(f"  Avg Collisions: {avg_collisions:.1f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
                print(f"  Q-table Size: {sum(len(q) for q in self.q_tables.values())}")
                print(f"  Best Completion Rate: {best_completion_rate:.1%}")
        
        training_time = time.time() - training_start_time
        if verbose:
            print(f"\nTraining completed in {training_time:.1f} seconds")
            print(f"Final epsilon: {self.epsilon:.3f}")
            print(f"Best completion rate achieved: {best_completion_rate:.1%}")
    
    def evaluate(self, num_episodes=20, max_steps_per_episode=300, verbose=True):
        """
        Evaluate trained Q-learning agents
        
        Returns:
            dict: Evaluation metrics
        """
        if verbose:
            print(f"Evaluating Q-Learning agents over {num_episodes} episodes...")
        
        evaluation_metrics = []
        episode_details = []
        
        # Temporarily disable exploration
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # Pure exploitation
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            episode_length = 0
            episode_rewards = {robot_id: 0 for robot_id in self.env.robots.keys()}
            
            # Get initial state representations
            state_reprs = {}
            for robot_id in self.env.robots.keys():
                state_reprs[robot_id] = self.get_state_representation(robot_id, state)
            
            done = False
            
            while not done and episode_length < max_steps_per_episode:
                # Choose actions (no exploration)
                actions = {}
                for robot_id in self.env.robots.keys():
                    actions[robot_id] = self.choose_action(robot_id, state_reprs[robot_id], training=False)
                
                # Execute actions
                next_state, rewards, done, info = self.env.step(actions)
                
                # Update state representations
                for robot_id in self.env.robots.keys():
                    state_reprs[robot_id] = self.get_state_representation(robot_id, next_state)
                    episode_rewards[robot_id] += rewards[robot_id]
                
                state = next_state
                episode_length += 1
            
            # Get final metrics
            metrics = self.env.get_metrics()
            evaluation_metrics.append(metrics)
            
            episode_details.append({
                'episode': episode + 1,
                'completion_rate': metrics['task_completion_rate'],
                'collisions': metrics['collision_count'],
                'steps': episode_length,
                'total_reward': sum(episode_rewards.values()),
                'path_efficiency': metrics['avg_path_efficiency']
            })
            
            if verbose:
                print(f"Episode {episode + 1}: "
                      f"Completion {metrics['task_completion_rate']:.1%}, "
                      f"Collisions {metrics['collision_count']}, "
                      f"Steps {episode_length}")
        
        # Restore epsilon
        self.epsilon = old_epsilon
        
        # Calculate aggregate statistics
        avg_metrics = {}
        for key in evaluation_metrics[0].keys():
            if isinstance(evaluation_metrics[0][key], (int, float)):
                values = [m[key] for m in evaluation_metrics]
                avg_metrics[f'mean_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
                avg_metrics[f'min_{key}'] = np.min(values)
                avg_metrics[f'max_{key}'] = np.max(values)
        
        # Success rate (completion rate > 80% and collisions < 5)
        success_episodes = sum(1 for m in evaluation_metrics 
                             if m['task_completion_rate'] > 0.8 and m['collision_count'] < 5)
        avg_metrics['success_rate'] = success_episodes / num_episodes
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"Success Rate: {avg_metrics['success_rate']:.1%}")
            print(f"Average Task Completion: {avg_metrics['mean_task_completion_rate']:.1%} ± {avg_metrics['std_task_completion_rate']:.1%}")
            print(f"Average Collisions: {avg_metrics['mean_collision_count']:.1f} ± {avg_metrics['std_collision_count']:.1f}")
            print(f"Average Path Efficiency: {avg_metrics['mean_avg_path_efficiency']:.3f} ± {avg_metrics['std_avg_path_efficiency']:.3f}")
            print(f"Average Steps: {avg_metrics['mean_total_steps']:.1f} ± {avg_metrics['std_total_steps']:.1f}")
        
        return avg_metrics, episode_details
    
    def plot_training_progress(self, save_path=None):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Q-Learning Training Progress', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.training_stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Task completion rates
        axes[0, 1].plot(self.training_stats['task_completion_rates'])
        axes[0, 1].set_title('Task Completion Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Completion Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True)
        
        # Collision counts
        axes[0, 2].plot(self.training_stats['collision_counts'])
        axes[0, 2].set_title('Collision Count')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Collisions')
        axes[0, 2].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(self.training_stats['episode_lengths'])
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Epsilon decay
        axes[1, 1].plot(self.training_stats['epsilon_values'])
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True)
        
        # Q-table growth
        axes[1, 2].plot(self.training_stats['q_table_sizes'])
        axes[1, 2].set_title('Q-Table Size')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Number of States')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """Save trained Q-tables and training stats"""
        model_data = {
            'q_tables': dict(self.q_tables),
            'training_stats': self.training_stats,
            'parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'current_epsilon': self.epsilon
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Q-Learning model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained Q-tables and training stats"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_tables = defaultdict(lambda: defaultdict(lambda: np.zeros(self.num_actions)))
        for robot_id, q_table in model_data['q_tables'].items():
            self.q_tables[robot_id] = q_table
        
        self.training_stats = model_data['training_stats']
        
        # Load parameters
        params = model_data['parameters']
        self.learning_rate = params['learning_rate']
        self.discount_factor = params['discount_factor']
        self.epsilon_start = params['epsilon_start']
        self.epsilon_end = params['epsilon_end']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon = params['current_epsilon']
        
        print(f"Q-Learning model loaded from {filepath}")


def compare_with_random_baseline(env: MultiRobotEnvironment, 
                               trained_qlearning: MultiRobotQLearning,
                               num_episodes=20):
    """
    Compare trained Q-learning performance with random baseline
    """
    print("Comparing Q-Learning vs Random Policy...")
    print("=" * 50)
    
    # Evaluate Q-Learning
    print("Evaluating Q-Learning Policy:")
    ql_metrics, ql_details = trained_qlearning.evaluate(num_episodes, verbose=True)
    
    # Evaluate Random Policy
    print(f"\nEvaluating Random Policy:")
    random_metrics, random_details = env.benchmark_random_policy(num_episodes)
    
    # Comparison
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    print(f"OBJECTIVE 1: Minimizing Travel Time & Path Length")
    print(f"  Q-Learning Avg Steps: {ql_metrics['mean_total_steps']:.1f}")
    print(f"  Random Avg Steps:     {random_metrics['avg_total_steps']:.1f}")
    print(f"  Improvement: {((random_metrics['avg_total_steps'] - ql_metrics['mean_total_steps']) / random_metrics['avg_total_steps'] * 100):.1f}%")
    
    print(f"\n  Q-Learning Path Efficiency: {ql_metrics['mean_avg_path_efficiency']:.3f}")
    print(f"  Random Path Efficiency:     {random_metrics['avg_avg_path_efficiency']:.3f}")
    print(f"  Improvement: {((ql_metrics['mean_avg_path_efficiency'] - random_metrics['avg_avg_path_efficiency']) / random_metrics['avg_avg_path_efficiency'] * 100):.1f}%")
    
    print(f"\nOBJECTIVE 2: Collision-Free Navigation")
    print(f"  Q-Learning Avg Collisions: {ql_metrics['mean_collision_count']:.1f}")
    print(f"  Random Avg Collisions:     {random_metrics['avg_collision_count']:.1f}")
    print(f"  Improvement: {((random_metrics['avg_collision_count'] - ql_metrics['mean_collision_count']) / max(random_metrics['avg_collision_count'], 1) * 100):.1f}%")
    
    print(f"\nOBJECTIVE 3: Task Allocation & Multi-Robot Coordination")
    print(f"  Q-Learning Completion Rate: {ql_metrics['mean_task_completion_rate']:.1%}")
    print(f"  Random Completion Rate:     {random_metrics['avg_task_completion_rate']:.1%}")
    print(f"  Improvement: {((ql_metrics['mean_task_completion_rate'] - random_metrics['avg_task_completion_rate']) / max(random_metrics['avg_task_completion_rate'], 0.01) * 100):.1f}%")
    
    print(f"\nOVERALL SUCCESS RATE")
    print(f"  Q-Learning Success Rate: {ql_metrics['success_rate']:.1%}")
    print(f"  Random Success Rate:     {random_metrics['success_rate']:.1%}")
    
    return ql_metrics, random_metrics


def main():
    """
    Main function to demonstrate Q-learning on multi-robot environment
    """
    print("Multi-Robot Q-Learning Demonstration")
    print("=" * 50)
    
    # Create environment
    env = MultiRobotEnvironment(
        grid_size=(10, 10),
        num_robots=3, 
        num_tasks=4,
        seed=42
    )
    
    print(f"Environment created: {env.grid_size} grid, {env.num_robots} robots, {env.num_tasks} tasks")
    
    # Show initial environment
    print("\nInitial Environment:")
    env.render(show_paths=False)
    
    # Initialize Q-learning
    qlearning = MultiRobotQLearning(
        env=env,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995
    )
    
    # Train the agents
    print("\nStarting Q-Learning training...")
    qlearning.train(
        num_episodes=1500,
        max_steps_per_episode=200,
        save_interval=300,
        verbose=True
    )
    
    # Plot training progress
    print("\nPlotting training progress...")
    qlearning.plot_training_progress("qlearning_training_progress.png")
    
    # Evaluate trained agents
    print("\nEvaluating trained Q-Learning agents...")
    eval_metrics, eval_details = qlearning.evaluate(num_episodes=30, verbose=True)
    
    # Save trained model
    qlearning.save_model("trained_qlearning_model.pkl")
    
    # Compare with random baseline
    print("\nRunning comparison with random policy...")
    ql_metrics, random_metrics = compare_with_random_baseline(env, qlearning, num_episodes=20)
    
    # Demonstrate trained agents
    print("\nDemonstrating trained Q-learning agents...")
    env.reset()
    
    # Run a demonstration episode
    state = env.get_state()
    done = False
    step_count = 0
    
    print("Running demonstration episode...")
    while not done and step_count < 150:
        # Get actions from trained Q-learning agents
        actions = {}
        for robot_id in env.robots.keys():
            state_repr = qlearning.get_state_representation(robot_id, state)
            actions[robot_id] = qlearning.choose_action(robot_id, state_repr, training=False)
        
        # Execute step
        state, rewards, done, info = env.step(actions)
        step_count += 1
        
        if step_count % 25 == 0:
            print(f"Step {step_count}: {info['tasks_completed']}/{info['total_tasks']} tasks completed")
    
    # Show final result
    print(f"\nDemonstration completed in {step_count} steps")
    final_metrics = env.get_metrics()
    print(f"Final demonstration metrics:")
    print(f"  Task completion rate: {final_metrics['task_completion_rate']:.1%}")
    print(f"  Collisions: {final_metrics['collision_count']}")
    print(f"  Path efficiency: {final_metrics['avg_path_efficiency']:.3f}")
    
    # Show final environment state
    print("\nFinal environment state:")
    env.render(show_paths=True)
    
    # Summary report
    print("\n" + "="*60)
    print("MULTI-ROBOT Q-LEARNING EVALUATION SUMMARY")
    print("="*60)
    
    print("\nOBJECTIVE ACHIEVEMENT ANALYSIS:")
    
    print(f"\n1. MINIMIZING TRAVEL TIME & PATH LENGTH:")
    print(f"   ✓ Average steps per episode: {eval_metrics['mean_total_steps']:.1f}")
    print(f"   ✓ Path efficiency: {eval_metrics['mean_avg_path_efficiency']:.3f}")
    if eval_metrics['mean_avg_path_efficiency'] > 0.3:
        print(f"   → GOOD: Robots learned efficient pathfinding")
    else:
        print(f"   → NEEDS IMPROVEMENT: Path efficiency could be better")
    
    print(f"\n2. COLLISION-FREE NAVIGATION:")
    print(f"   ✓ Average collisions per episode: {eval_metrics['mean_collision_count']:.1f}")
    print(f"   ✓ Collision rate: {eval_metrics['mean_collision_rate']:.3f}")
    if eval_metrics['mean_collision_count'] < 3:
        print(f"   → EXCELLENT: Low collision rate achieved")
    elif eval_metrics['mean_collision_count'] < 6:
        print(f"   → GOOD: Moderate collision avoidance")
    else:
        print(f"   → NEEDS IMPROVEMENT: High collision rate")
    
    print(f"\n3. TASK ALLOCATION & COORDINATION:")
    print(f"   ✓ Average task completion rate: {eval_metrics['mean_task_completion_rate']:.1%}")
    print(f"   ✓ Success rate: {eval_metrics['success_rate']:.1%}")
    if eval_metrics['mean_task_completion_rate'] > 0.8:
        print(f"   → EXCELLENT: High task completion achieved")
    elif eval_metrics['mean_task_completion_rate'] > 0.6:
        print(f"   → GOOD: Reasonable task completion")
    else:
        print(f"   → NEEDS IMPROVEMENT: Low task completion rate")
    
    print(f"\nOVERALL ASSESSMENT:")
    if eval_metrics['success_rate'] > 0.7:
        print(f"   🎯 STRONG PERFORMANCE: Q-Learning successfully learned multi-robot coordination")
    elif eval_metrics['success_rate'] > 0.4:
        print(f"   📈 MODERATE PERFORMANCE: Q-Learning showed learning but needs refinement")
    else:
        print(f"   ⚠️  WEAK PERFORMANCE: Q-Learning struggled with the multi-robot task")
    
    print(f"\nKEY INSIGHTS:")
    print(f"   • Training converged after ~{len(qlearning.training_stats['episode_rewards'])} episodes")
    print(f"   • Q-table size grew to {sum(len(q) for q in qlearning.q_tables.values())} states")
    print(f"   • Final exploration rate: {qlearning.epsilon:.3f}")
    
    improvements = []
    if eval_metrics['mean_collision_count'] > 5:
        improvements.append("Collision avoidance (consider larger penalties)")
    if eval_metrics['mean_task_completion_rate'] < 0.7:
        improvements.append("Task allocation strategy (better reward shaping)")
    if eval_metrics['mean_avg_path_efficiency'] < 0.3:
        improvements.append("Path planning efficiency (distance-based rewards)")
    
    if improvements:
        print(f"\nSUGGESTED IMPROVEMENTS:")
        for i, improvement in enumerate(improvements, 1):
            print(f"   {i}. {improvement}")
    
    print("\n" + "="*60)
    print("Evaluation complete! Check the generated plots and saved model.")


if __name__ == "__main__":
    main()