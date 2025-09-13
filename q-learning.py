import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque
import pickle
import json
from typing import Dict, List, Tuple, Optional
import time

# Import from your environment (assuming it's in the same directory)
from Environment.multi_robot_env import MultiRobotEnvironment

class QLearningAgent:
    """
    Q-Learning agent for multi-robot path planning with coordination mechanisms
    """
    
    def __init__(self, 
                 robot_id: int,
                 grid_size: Tuple[int, int],
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent
        
        Args:
            robot_id: Unique identifier for this robot
            grid_size: Size of the environment grid
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum epsilon value
        """
        self.robot_id = robot_id
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Action space: 0=stay, 1=up, 2=right, 3=down, 4=left
        self.num_actions = 5
        self.actions = list(range(self.num_actions))
        
        # Learning statistics
        self.training_episodes = 0
        self.cumulative_reward = 0
        self.episode_rewards = []
        self.collision_count = 0
        
        # Coordination memory
        self.other_robots_memory = {}
        self.coordination_weight = 0.2
        
    def get_state_key(self, observation: Dict) -> str:
        """
        Convert observation to a hashable state key for Q-table
        
        The state includes:
        - Robot's current position
        - Task information (target position, whether carrying)
        - Local obstacles and other robots
        - Distance to target
        """
        if observation is None:
            return "terminal"
        
        pos = observation['position']
        
        # Task information
        task_info = ""
        if observation['task_info']:
            target = observation['task_info']['target']
            carrying = observation['task_info']['carrying']
            distance = observation['task_info']['distance']
            task_info = f"t_{target[0]}_{target[1]}_c{int(carrying)}_d{min(distance, 20)}"
        else:
            task_info = "no_task"
        
        # Local environment (simplified to reduce state space)
        local_obs = len(observation['local_obstacles'])
        local_robots = len(observation['local_robots'])
        
        # Create compact state representation
        state_key = f"p_{pos[0]}_{pos[1]}_{task_info}_o{local_obs}_r{local_robots}"
        
        return state_key
    
    def choose_action(self, observation: Dict, other_robots_positions: List[Tuple] = None) -> int:
        """
        Choose action using epsilon-greedy policy with coordination considerations
        """
        if observation is None:
            return 0  # Stay action for terminal states
        
        state_key = self.get_state_key(observation)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.choice(self.actions)
        else:
            # Exploitation: best action from Q-table with coordination
            q_values = self.q_table[state_key]
            
            if not q_values:
                # If no Q-values yet, choose random action
                action = random.choice(self.actions)
            else:
                # Get action with highest Q-value, considering coordination
                coordinated_q_values = self._apply_coordination_bonus(
                    observation, q_values, other_robots_positions)
                action = max(coordinated_q_values.keys(), 
                           key=coordinated_q_values.get)
        
        return action
    
    def _apply_coordination_bonus(self, observation: Dict, q_values: Dict, 
                                other_robots_positions: List[Tuple] = None) -> Dict:
        """
        Apply coordination bonus/penalty to Q-values to avoid conflicts
        """
        if not other_robots_positions:
            return q_values
        
        current_pos = observation['position']
        coordinated_q_values = q_values.copy()
        
        # Actions: 0=stay, 1=up, 2=right, 3=down, 4=left
        action_moves = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        
        for action, q_value in q_values.items():
            dx, dy = action_moves[action]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check if this action would lead to collision with other robots
            collision_penalty = 0
            for other_pos in other_robots_positions:
                if next_pos == other_pos:
                    collision_penalty += 10  # Strong penalty for direct collision
                elif self._manhattan_distance(next_pos, other_pos) <= 1:
                    collision_penalty += 2   # Mild penalty for getting too close
            
            # Apply coordination adjustment
            coordinated_q_values[action] = q_value - (collision_penalty * self.coordination_weight)
        
        return coordinated_q_values
    
    def update_q_value(self, state: str, action: int, reward: float, 
                      next_state: str, done: bool):
        """
        Update Q-value using Q-learning update rule
        """
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Best next action Q-value
        if done:
            max_next_q = 0
        else:
            next_q_values = self.q_table[next_state]
            max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
        
        # Update statistics
        self.cumulative_reward += reward
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _manhattan_distance(self, pos1: Tuple, pos2: Tuple) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def save_model(self, filepath: str):
        """Save the Q-table and agent parameters"""
        model_data = {
            'robot_id': self.robot_id,
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'training_episodes': self.training_episodes,
            'episode_rewards': self.episode_rewards,
            'collision_count': self.collision_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved for robot {self.robot_id} to {filepath}")
    
    def load_model(self, filepath: str):
        """Load Q-table and agent parameters"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
        self.epsilon = model_data['epsilon']
        self.training_episodes = model_data['training_episodes']
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.collision_count = model_data.get('collision_count', 0)
        
        print(f"Model loaded for robot {self.robot_id} from {filepath}")


class MultiRobotQLearningCoordinator:
    """
    Coordinator for multiple Q-learning agents with advanced coordination mechanisms
    """
    
    def __init__(self, 
                 environment: MultiRobotEnvironment,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize multi-robot Q-learning coordinator
        """
        self.env = environment
        self.num_robots = environment.num_robots
        self.grid_size = environment.grid_size
        
        # Create Q-learning agents for each robot
        self.agents = {}
        for robot_id in range(self.num_robots):
            self.agents[robot_id] = QLearningAgent(
                robot_id=robot_id,
                grid_size=self.grid_size,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            )
        
        # Training statistics
        self.episode_count = 0
        self.training_history = []
        self.performance_metrics = []
        
        # Coordination mechanisms
        self.communication_enabled = True
        self.task_auction_enabled = True
        
    def train(self, episodes: int = 1000, verbose: bool = True, 
              save_interval: int = 100):
        """
        Train all agents using Q-learning with coordination
        
        Args:
            episodes: Number of training episodes
            verbose: Whether to print training progress
            save_interval: Save models every N episodes
        """
        print(f"Starting Q-Learning training for {self.num_robots} robots")
        print(f"Training for {episodes} episodes...")
        print("="*60)
        
        episode_rewards_history = []
        
        for episode in range(episodes):
            # Reset environment for new episode
            state = self.env.reset()
            
            # Get initial observations for all robots
            observations = {}
            for robot_id in self.agents.keys():
                observations[robot_id] = self.env.get_robot_observation(robot_id)
            
            # Store initial states
            states = {}
            for robot_id, agent in self.agents.items():
                states[robot_id] = agent.get_state_key(observations[robot_id])
            
            episode_rewards = {robot_id: 0 for robot_id in self.agents.keys()}
            step_count = 0
            max_steps = self.env.max_steps
            
            while step_count < max_steps:
                # Get actions from all agents with coordination
                actions = self._get_coordinated_actions(observations)
                
                # Execute actions in environment
                next_state, rewards, done, info = self.env.step(actions)
                
                # Get next observations
                next_observations = {}
                for robot_id in self.agents.keys():
                    next_observations[robot_id] = self.env.get_robot_observation(robot_id)
                
                # Update Q-values for all agents
                for robot_id, agent in self.agents.items():
                    current_state = states[robot_id]
                    action = actions[robot_id]
                    reward = rewards[robot_id]
                    next_state_key = agent.get_state_key(next_observations[robot_id])
                    
                    # Enhanced reward shaping for better learning
                    shaped_reward = self._shape_reward(
                        robot_id, reward, observations[robot_id], 
                        next_observations[robot_id], info)
                    
                    agent.update_q_value(current_state, action, shaped_reward, 
                                       next_state_key, done)
                    
                    episode_rewards[robot_id] += shaped_reward
                
                # Update states and observations
                states = {robot_id: agent.get_state_key(next_observations[robot_id]) 
                         for robot_id, agent in self.agents.items()}
                observations = next_observations
                step_count += 1
                
                if done:
                    break
            
            # Decay exploration rates
            for agent in self.agents.values():
                agent.decay_epsilon()
                agent.training_episodes += 1
            
            # Record episode statistics
            total_episode_reward = sum(episode_rewards.values())
            episode_rewards_history.append(total_episode_reward)
            
            # Get episode metrics
            metrics = self.env.get_metrics()
            self.performance_metrics.append(metrics)
            
            # Update training history
            self.training_history.append({
                'episode': episode,
                'total_reward': total_episode_reward,
                'individual_rewards': episode_rewards.copy(),
                'metrics': metrics,
                'epsilon': self.agents[0].epsilon  # All agents have same epsilon
            })
            
            # Verbose output
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards_history[-50:])
                success_rate = np.mean([m['success'] for m in self.performance_metrics[-50:]])
                avg_completion = np.mean([m['task_completion_rate'] for m in self.performance_metrics[-50:]])
                avg_collisions = np.mean([m['collision_count'] for m in self.performance_metrics[-50:]])
                
                print(f"Episode {episode + 1:4d}: "
                      f"Avg Reward: {avg_reward:8.2f}, "
                      f"Success Rate: {success_rate:.2%}, "
                      f"Completion: {avg_completion:.2%}, "
                      f"Collisions: {avg_collisions:.1f}, "
                      f"Epsilon: {self.agents[0].epsilon:.3f}")
            
            # Save models periodically
            if (episode + 1) % save_interval == 0:
                self.save_models(f"models_episode_{episode + 1}")
            
            self.episode_count += 1
        
        print("\nTraining completed!")
        self._print_training_summary()
    
    def _get_coordinated_actions(self, observations: Dict) -> Dict[int, int]:
        """
        Get actions for all robots with coordination mechanisms
        """
        actions = {}
        
        # Get current positions for coordination
        current_positions = []
        for robot_id, obs in observations.items():
            if obs:
                current_positions.append(obs['position'])
            else:
                current_positions.append(None)
        
        # Get actions from each agent
        for robot_id, agent in self.agents.items():
            if observations[robot_id] is not None:
                # Get positions of other robots
                other_positions = [pos for i, pos in enumerate(current_positions) 
                                 if i != robot_id and pos is not None]
                
                action = agent.choose_action(observations[robot_id], other_positions)
                actions[robot_id] = action
            else:
                actions[robot_id] = 0  # Stay action if no observation
        
        # Apply additional coordination rules
        actions = self._resolve_action_conflicts(actions, observations)
        
        return actions
    
    def _resolve_action_conflicts(self, actions: Dict[int, int], 
                                observations: Dict) -> Dict[int, int]:
        """
        Resolve potential conflicts between robot actions
        """
        # Actions: 0=stay, 1=up, 2=right, 3=down, 4=left
        action_moves = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Calculate intended next positions
        intended_positions = {}
        for robot_id, action in actions.items():
            if observations[robot_id] is not None:
                current_pos = observations[robot_id]['position']
                dx, dy = action_moves[action]
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                intended_positions[robot_id] = next_pos
        
        # Check for conflicts and resolve them
        resolved_actions = actions.copy()
        position_conflicts = {}
        
        # Find robots intending to move to the same position
        for robot_id, next_pos in intended_positions.items():
            if next_pos not in position_conflicts:
                position_conflicts[next_pos] = []
            position_conflicts[next_pos].append(robot_id)
        
        # Resolve conflicts by having lower-priority robots stay
        for next_pos, conflicting_robots in position_conflicts.items():
            if len(conflicting_robots) > 1:
                # Priority based on robot ID and current task status
                def get_priority(robot_id):
                    obs = observations[robot_id]
                    if obs and obs['task_info']:
                        if obs['task_info']['carrying']:
                            return 10  # Highest priority for carrying robots
                        else:
                            return 5   # Medium priority for assigned robots
                    return robot_id    # Default priority by ID
                
                # Sort by priority (higher priority first)
                conflicting_robots.sort(key=get_priority, reverse=True)
                
                # Keep highest priority robot's action, others stay
                for i, robot_id in enumerate(conflicting_robots):
                    if i > 0:  # Not the highest priority
                        resolved_actions[robot_id] = 0  # Stay action
        
        return resolved_actions
    
    def _shape_reward(self, robot_id: int, base_reward: float, 
                     current_obs: Dict, next_obs: Dict, info: Dict) -> float:
        """
        Apply reward shaping to improve learning efficiency
        """
        shaped_reward = base_reward
        
        if current_obs is None or next_obs is None:
            return shaped_reward
        
        # Reward for moving towards task objectives
        if (current_obs.get('task_info') and next_obs.get('task_info') and
            current_obs['task_info'] is not None and next_obs['task_info'] is not None):
            
            current_distance = current_obs['task_info']['distance']
            next_distance = next_obs['task_info']['distance']
            
            # Reward for getting closer to target
            if next_distance < current_distance:
                shaped_reward += 2  # Progress bonus
            elif next_distance > current_distance:
                shaped_reward -= 1  # Regress penalty
        
        # Penalty for excessive collisions
        if info.get('collisions_this_step', False):
            shaped_reward -= 10
        
        # Reward for efficient movement (not staying idle unless necessary)
        if (current_obs['position'] == next_obs['position'] and 
            current_obs.get('task_info') is not None):
            shaped_reward -= 0.5  # Small penalty for unnecessary idling
        
        # Bonus for coordination (staying away from other robots when not necessary)
        if len(next_obs.get('local_robots', [])) == 0:
            shaped_reward += 0.5  # Small bonus for avoiding crowding
        
        return shaped_reward
    
    def evaluate(self, episodes: int = 10, render: bool = False) -> Dict:
        """
        Evaluate the trained agents
        """
        print(f"Evaluating trained agents over {episodes} episodes...")
        
        evaluation_results = []
        
        # Temporarily disable exploration
        original_epsilons = {}
        for robot_id, agent in self.agents.items():
            original_epsilons[robot_id] = agent.epsilon
            agent.epsilon = 0.0  # Pure exploitation
        
        for episode in range(episodes):
            # Reset environment
            state = self.env.reset()
            
            # Get initial observations
            observations = {}
            for robot_id in self.agents.keys():
                observations[robot_id] = self.env.get_robot_observation(robot_id)
            
            episode_rewards = {robot_id: 0 for robot_id in self.agents.keys()}
            step_count = 0
            
            while step_count < self.env.max_steps:
                # Get actions (no exploration)
                actions = self._get_coordinated_actions(observations)
                
                # Execute actions
                next_state, rewards, done, info = self.env.step(actions)
                
                # Update observations
                for robot_id in self.agents.keys():
                    observations[robot_id] = self.env.get_robot_observation(robot_id)
                    episode_rewards[robot_id] += rewards[robot_id]
                
                step_count += 1
                
                if done:
                    break
            
            # Get episode metrics
            metrics = self.env.get_metrics()
            metrics['episode_rewards'] = episode_rewards
            metrics['total_reward'] = sum(episode_rewards.values())
            evaluation_results.append(metrics)
            
            if render and episode == 0:  # Render first episode
                print(f"\nEvaluation Episode {episode + 1}")
                self.env.render(show_paths=True)
        
        # Restore original exploration rates
        for robot_id, agent in self.agents.items():
            agent.epsilon = original_epsilons[robot_id]
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(evaluation_results)
        
        return aggregate_metrics
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate evaluation metrics"""
        metrics = {}
        
        # Calculate averages and standard deviations
        numerical_keys = ['task_completion_rate', 'collision_count', 'avg_path_length',
                         'total_distance', 'throughput', 'avg_path_efficiency', 'total_steps']
        
        for key in numerical_keys:
            values = [r[key] for r in results if key in r]
            if values:
                metrics[f'avg_{key}'] = np.mean(values)
                metrics[f'std_{key}'] = np.std(values)
        
        # Success rate
        success_count = sum(1 for r in results if r.get('success', False))
        metrics['success_rate'] = success_count / len(results)
        
        # Reward statistics
        total_rewards = [r['total_reward'] for r in results]
        metrics['avg_total_reward'] = np.mean(total_rewards)
        metrics['std_total_reward'] = np.std(total_rewards)
        
        return metrics
    
    def _print_training_summary(self):
        """Print comprehensive training summary"""
        if not self.training_history:
            return
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        # Recent performance (last 100 episodes)
        recent_episodes = self.training_history[-100:]
        
        avg_reward = np.mean([ep['total_reward'] for ep in recent_episodes])
        success_rate = np.mean([ep['metrics']['success'] for ep in recent_episodes])
        avg_completion = np.mean([ep['metrics']['task_completion_rate'] for ep in recent_episodes])
        avg_collisions = np.mean([ep['metrics']['collision_count'] for ep in recent_episodes])
        final_epsilon = recent_episodes[-1]['epsilon']
        
        print(f"Episodes Trained: {self.episode_count}")
        print(f"Final Epsilon: {final_epsilon:.4f}")
        print(f"\nRecent Performance (last {len(recent_episodes)} episodes):")
        print(f"  Average Total Reward: {avg_reward:.2f}")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Task Completion Rate: {avg_completion:.2%}")
        print(f"  Average Collisions: {avg_collisions:.2f}")
        
        print(f"\nObjectives Achievement:")
        print(f"  1. Travel Time & Path Length: {'✓' if avg_completion > 0.7 else '✗'} "
              f"({avg_completion:.1%} completion rate)")
        print(f"  2. Collision-Free Navigation: {'✓' if avg_collisions < 3 else '✗'} "
              f"({avg_collisions:.1f} avg collisions)")
        print(f"  3. Multi-Robot Coordination: {'✓' if success_rate > 0.5 else '✗'} "
              f"({success_rate:.1%} success rate)")
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress and metrics"""
        if not self.training_history:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Q-Learning Training Progress', fontsize=16)
        
        episodes = [ep['episode'] for ep in self.training_history]
        
        # Total rewards
        total_rewards = [ep['total_reward'] for ep in self.training_history]
        axes[0, 0].plot(episodes, total_rewards, alpha=0.7)
        axes[0, 0].plot(episodes, self._smooth(total_rewards, 50), 'r-', linewidth=2)
        axes[0, 0].set_title('Total Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Task completion rate
        completion_rates = [ep['metrics']['task_completion_rate'] for ep in self.training_history]
        axes[0, 1].plot(episodes, completion_rates, alpha=0.7)
        axes[0, 1].plot(episodes, self._smooth(completion_rates, 50), 'g-', linewidth=2)
        axes[0, 1].set_title('Task Completion Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Completion Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True)
        
        # Collision count
        collisions = [ep['metrics']['collision_count'] for ep in self.training_history]
        axes[0, 2].plot(episodes, collisions, alpha=0.7)
        axes[0, 2].plot(episodes, self._smooth(collisions, 50), 'r-', linewidth=2)
        axes[0, 2].set_title('Collision Count')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Collisions')
        axes[0, 2].grid(True)
        
        # Success rate
        success_rate_window = 100
        success_rates = []
        for i in range(len(self.training_history)):
            start_idx = max(0, i - success_rate_window + 1)
            window_episodes = self.training_history[start_idx:i+1]
            success_rate = np.mean([ep['metrics']['success'] for ep in window_episodes])
            success_rates.append(success_rate)
        
        axes[1, 0].plot(episodes, success_rates, 'purple', linewidth=2)
        axes[1, 0].set_title(f'Success Rate (rolling {success_rate_window} episodes)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True)
        
        # Average path length
        path_lengths = [ep['metrics']['avg_path_length'] for ep in self.training_history]
        axes[1, 1].plot(episodes, path_lengths, alpha=0.7)
        axes[1, 1].plot(episodes, self._smooth(path_lengths, 50), 'orange', linewidth=2)
        axes[1, 1].set_title('Average Path Length')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Path Length')
        axes[1, 1].grid(True)
        
        # Epsilon decay
        epsilons = [ep['epsilon'] for ep in self.training_history]
        axes[1, 2].plot(episodes, epsilons, 'brown', linewidth=2)
        axes[1, 2].set_title('Exploration Rate (Epsilon)')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Epsilon')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {save_path}")
        
        plt.show()
    
    def _smooth(self, data: List[float], window: int) -> List[float]:
        """Apply moving average smoothing"""
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            smoothed.append(np.mean(data[start_idx:i+1]))
        return smoothed
    
    def save_models(self, directory: str):
        """Save all agent models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for robot_id, agent in self.agents.items():
            filepath = os.path.join(directory, f"robot_{robot_id}_model.pkl")
            agent.save_model(filepath)
        
        # Save coordinator metadata
        metadata = {
            'episode_count': self.episode_count,
            'training_history': self.training_history[-100:],  # Keep last 100 episodes
            'num_robots': self.num_robots,
            'grid_size': self.grid_size
        }
        
        with open(os.path.join(directory, "coordinator_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"All models saved to {directory}/")
    
    def load_models(self, directory: str):
        """Load all agent models"""
        import os
        
        for robot_id, agent in self.agents.items():
            filepath = os.path.join(directory, f"robot_{robot_id}_model.pkl")
            if os.path.exists(filepath):
                agent.load_model(filepath)
        
        # Load coordinator metadata if available
        metadata_path = os.path.join(directory, "coordinator_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.episode_count = metadata.get('episode_count', 0)
            self.training_history = metadata.get('training_history', [])
        
        print(f"All models loaded from {directory}/")
    
    def run_demo(self, episodes: int = 3, render: bool = True, save_animation: bool = False):
        """
        Run a demonstration of the trained agents
        """
        print(f"Running demonstration for {episodes} episodes...")
        
        # Set to exploitation mode
        original_epsilons = {}
        for robot_id, agent in self.agents.items():
            original_epsilons[robot_id] = agent.epsilon
            agent.epsilon = 0.0  # Pure exploitation
        
        for episode in range(episodes):
            print(f"\n{'='*50}")
            print(f"DEMONSTRATION EPISODE {episode + 1}")
            print(f"{'='*50}")
            
            # Reset environment
            state = self.env.reset()
            
            if render:
                print("Initial State:")
                self.env.render(show_paths=False)
            
            # Get initial observations
            observations = {}
            for robot_id in self.agents.keys():
                observations[robot_id] = self.env.get_robot_observation(robot_id)
            
            step_count = 0
            episode_rewards = {robot_id: 0 for robot_id in self.agents.keys()}
            
            while step_count < self.env.max_steps:
                # Get coordinated actions
                actions = self._get_coordinated_actions(observations)
                
                # Execute step
                next_state, rewards, done, info = self.env.step(actions)
                
                # Update observations and rewards
                for robot_id in self.agents.keys():
                    observations[robot_id] = self.env.get_robot_observation(robot_id)
                    episode_rewards[robot_id] += rewards[robot_id]
                
                step_count += 1
                
                # Print step information
                if step_count % 20 == 0 or done:
                    print(f"\nStep {step_count}:")
                    print(f"  Actions: {[(rid, ['Stay', 'Up', 'Right', 'Down', 'Left'][action]) for rid, action in actions.items()]}")
                    print(f"  Tasks completed: {info['tasks_completed']}/{info['total_tasks']}")
                    if info['collisions_this_step']:
                        print(f"  ⚠️ Collision occurred!")
                
                if done:
                    break
            
            # Show final results
            metrics = self.env.get_metrics()
            
            print(f"\nEpisode {episode + 1} Results:")
            print(f"  Task Completion Rate: {metrics['task_completion_rate']:.1%}")
            print(f"  Total Collisions: {metrics['collision_count']}")
            print(f"  Average Path Length: {metrics['avg_path_length']:.2f}")
            print(f"  Total Steps: {step_count}")
            print(f"  Success: {'✅' if metrics['success'] else '❌'}")
            print(f"  Episode Rewards: {episode_rewards}")
            
            if render:
                print("\nFinal State:")
                self.env.render(show_paths=True)
            
            if save_animation and episode == 0:
                self.env.create_animation("qlearning_demo.gif")
        
        # Restore exploration rates
        for robot_id, agent in self.agents.items():
            agent.epsilon = original_epsilons[robot_id]


def main():
    """
    Main function to demonstrate Q-learning implementation
    """
    print("Multi-Robot Q-Learning Path Planning Implementation")
    print("="*60)
    
    # Environment parameters
    GRID_SIZE = (10, 10)
    NUM_ROBOTS = 3
    NUM_TASKS = 4
    SEED = 42
    
    # Q-learning hyperparameters
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EPSILON = 1.0
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.01
    
    # Training parameters
    TRAINING_EPISODES = 800
    EVALUATION_EPISODES = 10
    
    print(f"\nEnvironment Configuration:")
    print(f"  Grid Size: {GRID_SIZE}")
    print(f"  Robots: {NUM_ROBOTS}")
    print(f"  Tasks: {NUM_TASKS}")
    print(f"  Seed: {SEED}")
    
    print(f"\nQ-Learning Hyperparameters:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Discount Factor: {DISCOUNT_FACTOR}")
    print(f"  Initial Epsilon: {EPSILON}")
    print(f"  Epsilon Decay: {EPSILON_DECAY}")
    print(f"  Min Epsilon: {EPSILON_MIN}")
    
    # Create environment
    env = MultiRobotEnvironment(
        grid_size=GRID_SIZE,
        num_robots=NUM_ROBOTS,
        num_tasks=NUM_TASKS,
        seed=SEED
    )
    
    # Create Q-learning coordinator
    coordinator = MultiRobotQLearningCoordinator(
        environment=env,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN
    )
    
    # Show initial environment
    print("\nInitial Environment:")
    env.render(show_paths=False)
    
    print("\nStarting Training Phase...")
    
    # Train the agents
    coordinator.train(
        episodes=TRAINING_EPISODES,
        verbose=True,
        save_interval=200
    )
    
    # Plot training progress
    coordinator.plot_training_progress("training_progress.png")
    
    # Evaluate trained agents
    print("\nStarting Evaluation Phase...")
    eval_results = coordinator.evaluate(
        episodes=EVALUATION_EPISODES,
        render=False
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"Success Rate: {eval_results['success_rate']:.1%}")
    print(f"Average Task Completion: {eval_results['avg_task_completion_rate']:.1%} ± {eval_results['std_task_completion_rate']:.1%}")
    print(f"Average Collisions: {eval_results['avg_collision_count']:.2f} ± {eval_results['std_collision_count']:.2f}")
    print(f"Average Path Length: {eval_results['avg_avg_path_length']:.2f} ± {eval_results['std_avg_path_length']:.2f}")
    print(f"Average Total Reward: {eval_results['avg_total_reward']:.2f} ± {eval_results['std_total_reward']:.2f}")
    print(f"Path Efficiency: {eval_results['avg_avg_path_efficiency']:.3f} ± {eval_results['std_avg_path_efficiency']:.3f}")
    
    # Objectives assessment
    print(f"\n{'='*60}")
    print("OBJECTIVES ACHIEVEMENT ASSESSMENT")
    print("="*60)
    
    # Objective 1: Minimizing Travel Time & Path Length
    path_efficiency = eval_results['avg_avg_path_efficiency']
    completion_rate = eval_results['avg_task_completion_rate']
    obj1_score = (path_efficiency + completion_rate) / 2
    
    print(f"1. Minimizing Travel Time & Path Length:")
    print(f"   ✓ Task Completion Rate: {completion_rate:.1%}")
    print(f"   ✓ Path Efficiency: {path_efficiency:.3f}")
    print(f"   Overall Score: {obj1_score:.3f} {'✅' if obj1_score > 0.6 else '⚠️'}")
    
    # Objective 2: Collision-Free Navigation
    collision_rate = eval_results['avg_collision_count'] / eval_results['avg_total_steps']
    collision_free_episodes = eval_results['success_rate']  # Success implies low collisions
    
    print(f"\n2. Collision-Free Navigation in Dynamic Environments:")
    print(f"   ✓ Average Collisions per Episode: {eval_results['avg_collision_count']:.2f}")
    print(f"   ✓ Collision Rate per Step: {collision_rate:.4f}")
    print(f"   ✓ Success Rate (low collisions): {collision_free_episodes:.1%}")
    print(f"   Achievement: {'✅' if eval_results['avg_collision_count'] < 3.0 else '⚠️'}")
    
    # Objective 3: Task Allocation & Multi-Robot Coordination
    throughput = eval_results.get('avg_throughput', 0)
    coordination_score = eval_results['success_rate'] * completion_rate
    
    print(f"\n3. Task Allocation & Multi-Robot Coordination:")
    print(f"   ✓ Success Rate: {eval_results['success_rate']:.1%}")
    print(f"   ✓ System Throughput: {throughput:.4f} tasks/step")
    print(f"   ✓ Coordination Score: {coordination_score:.3f}")
    print(f"   Achievement: {'✅' if coordination_score > 0.5 else '⚠️'}")
    
    # Overall assessment
    overall_score = (obj1_score + (1 - min(collision_rate * 100, 1)) + coordination_score) / 3
    print(f"\n{'='*60}")
    print(f"OVERALL PERFORMANCE SCORE: {overall_score:.3f}")
    print(f"Rating: {get_performance_rating(overall_score)}")
    print("="*60)
    
    # Save models
    coordinator.save_models("trained_models")
    
    # Run demonstration
    print("\nRunning Trained Agent Demonstration...")
    coordinator.run_demo(episodes=2, render=True, save_animation=True)
    
    return coordinator, eval_results


def get_performance_rating(score: float) -> str:
    """Get performance rating based on score"""
    if score >= 0.8:
        return "Excellent ⭐⭐⭐⭐⭐"
    elif score >= 0.7:
        return "Good ⭐⭐⭐⭐"
    elif score >= 0.6:
        return "Satisfactory ⭐⭐⭐"
    elif score >= 0.5:
        return "Needs Improvement ⭐⭐"
    else:
        return "Poor ⭐"


def compare_with_baseline():
    """
    Compare Q-learning performance with random baseline
    """
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    # Create fresh environment for baseline
    env_baseline = MultiRobotEnvironment(
        grid_size=(10, 10),
        num_robots=3,
        num_tasks=4,
        seed=42
    )
    
    # Run baseline (random policy)
    baseline_metrics, _ = env_baseline.benchmark_random_policy(episodes=10)
    
    print("Random Policy Baseline:")
    print(f"  Success Rate: {baseline_metrics['success_rate']:.1%}")
    print(f"  Avg Completion: {baseline_metrics['avg_task_completion_rate']:.1%}")
    print(f"  Avg Collisions: {baseline_metrics['avg_collision_count']:.2f}")
    
    return baseline_metrics


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Run main training and evaluation
        coordinator, eval_results = main()
        
        # Compare with baseline
        baseline_results = compare_with_baseline()
        
        print(f"\n{'='*60}")
        print("FINAL COMPARISON: Q-Learning vs Random Policy")
        print("="*60)
        
        print("Metric                    | Q-Learning | Random   | Improvement")
        print("-" * 60)
        print(f"Success Rate             | {eval_results['success_rate']:8.1%} | {baseline_results['success_rate']:6.1%} | {eval_results['success_rate']/max(baseline_results['success_rate'], 0.01):6.1f}x")
        print(f"Task Completion          | {eval_results['avg_task_completion_rate']:8.1%} | {baseline_results['avg_task_completion_rate']:6.1%} | {eval_results['avg_task_completion_rate']/max(baseline_results['avg_task_completion_rate'], 0.01):6.1f}x")
        print(f"Collisions (lower=better)| {eval_results['avg_collision_count']:8.2f} | {baseline_results['avg_collision_count']:6.2f} | {baseline_results['avg_collision_count']/max(eval_results['avg_collision_count'], 0.01):6.1f}x better")
        
        print("\n🎉 Q-Learning implementation completed successfully!")
        print("📁 Models saved in 'trained_models/' directory")
        print("📊 Training progress plot saved as 'training_progress.png'")
        print("🎬 Demo animation saved as 'qlearning_demo.gif'")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure 'multi_robot_env.py' is in the same directory")
        print("or adjust the import path accordingly.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()