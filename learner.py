# learner.py
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class DQN(nn.Module):
    """Simple DQN Network."""

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNLearner:
    """DQN-based Reinforcement Learning Agent."""

    def __init__(self, rl_method, network, num_steps, learning_rate, balance,
                 num_epoches, discount_factor, start_epsilon, reuse_model,
                 output_path, stock_code, chart_data, training_data,
                 min_trading_price, max_trading_price, value_network_path):
        self.rl_method = rl_method
        self.network = network
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.balance = balance
        self.num_epoches = num_epoches
        self.discount_factor = discount_factor
        self.epsilon = start_epsilon
        self.reuse_model = reuse_model
        self.output_path = output_path
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.training_data = training_data
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price
        self.value_network_path = value_network_path

        # Initialize the DQN network
        input_dim = training_data.shape[1]  # Number of features
        output_dim = 3  # Actions: Buy, Hold, Sell
        self.model = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = []
        self.memory_size = 2000
        self.batch_size = 32

    def run(self, learning=True):
        """Runs the agent for training or testing."""
        for epoch in range(self.num_epoches):
            total_reward = 0
            state = self._get_initial_state()

            for t in range(len(self.training_data) - self.num_steps):
                action = self._select_action(state)
                reward, next_state = self._step(action)
                total_reward += reward

                # Store in memory for replay
                self._remember(state, action, reward, next_state)

                # Learn from experience if in training mode
                if learning:
                    self._learn_from_experience()

                # Move to the next state
                state = next_state

            print(f"Epoch {epoch + 1}/{self.num_epoches} - Total Reward: {total_reward}")

    def _get_initial_state(self):
        """Gets the initial state for the agent."""
        return self.training_data[:self.num_steps]

    def _select_action(self, state):
        """Selects an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Random action: Buy, Hold, Sell
        else:
            # Convert state DataFrame to a NumPy array
            state_tensor = torch.FloatTensor(state.values).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def _step(self, action):
        """Executes the given action and returns the reward and next state."""
        # Example reward function
        reward = random.random()  # Placeholder for reward logic

        # Fetch the next state based on the current timestep
        next_state = self.training_data[self.num_steps:self.num_steps * 2].reset_index(drop=True)
        return reward, next_state

    def _remember(self, state, action, reward, next_state):
        """Stores the experience in replay memory."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state))

    def _learn_from_experience(self):
        """Updates the model using experiences from the replay memory."""
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            # Convert state and next_state DataFrames to NumPy arrays
            state_tensor = torch.FloatTensor(state.values).unsqueeze(0)  # Shape: (1, num_features)
            next_state_tensor = torch.FloatTensor(next_state.values).unsqueeze(0)  # Shape: (1, num_features)
            reward_tensor = torch.FloatTensor([reward])  # Shape: (1)

            # Compute target
            with torch.no_grad():
                # Get the Q-values for the next state
                next_q_values = self.model(next_state_tensor)  # Shape: (1, 3)
                target = reward_tensor + self.discount_factor * torch.max(next_q_values)  # Shape: (1)

            # Get current prediction
            current_q = self.model(state_tensor)[0, action]  # Shape: ()

            # Update the model
            # Reshape target to match the input size
            loss = self.criterion(current_q.unsqueeze(0), target.unsqueeze(0))  # Ensure both have shape (1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_models(self):
        """Saves the trained model to disk."""
        model_path = os.path.join(self.output_path, f"{self.stock_code}_model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def predict(self):
        """Performs prediction using the trained model."""
        model_path = os.path.join(self.output_path, f"{self.stock_code}_model.pth")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode

        # Get initial state for prediction
        state = self._get_initial_state()

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state.values).unsqueeze(0)

        # Get predicted action
        with torch.no_grad():
            q_values = self.model(state_tensor)

        # Select action based on highest Q-value
        action = torch.argmax(q_values).item()
        print(f"Predicted action: {action}")