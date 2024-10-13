import os
import logging
import abc
import collections
import threading
import time
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm


#강화학습 기본클래스
class ReinforcementLearner(metaclass=abc.ABCMeta):

    def __init__(self, state_size, action_size, lr, discount_factor, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.discount_factor = discount_factor

        # 네트워크, 옵티마이저 설정
        self.network = self._build_network()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    #강화학습 알고리즘별 신경망 정의 추상메소드
    @abc.abstractmethod
    def _build_network(self):
        pass

    #강화학습 학습 진행 추상메소드
    @abc.abstractmethod
    def train(self, X_train, y_train):

        pass

    #알고리즘별 업데이트 방식을 정의하는 추상 메서드
    @abc.abstractmethod
    def update(self, state, action, reward, next_state, done):

        pass

    def save_model(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load_model(self, filename):
        self.network.load_state_dict(torch.load(filename))


# DQN 클래스
class DQNLearner(ReinforcementLearner):
    def __init__(self, X_train, y_train, X_test, y_test,lr, discount_factor):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


        self.state_size = X_train.shape[1] * X_train.shape[2]
        self.action_size = 3

        super().__init__(self.state_size, self.action_size, lr, discount_factor)

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def train(self, num_epochs):
        memory = collections.deque(maxlen=10000)
        batch_size = 64

        for epoch in tqdm(range(num_epochs)):
            state = self.X_train[0]  # 초기 상태 설정
            total_reward = 0  # 총 보상 초기화

            for t in range(len(self.X_train) - 1):  # 범위를 len(X_train) - 1로 수정
                action = self.select_action(state)
                next_state = self.X_train[t + 1]
                reward = self.y_train[t]

                # 메모리에 (상태, 행동, 보상, 다음 상태, 종료 여부) 추가
                memory.append((state, action, reward, next_state, False))

                # 배치가 준비되면 업데이트
                if len(memory) >= batch_size:
                    self.update_batch(memory, batch_size)

                # 상태 업데이트
                state = next_state
                total_reward += reward  # 보상 업데이트

            # 에폭마다 로그 출력
            print(f'Epoch {epoch + 1}/{num_epochs}, Total Reward: {total_reward}')

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)  # 랜덤 행동 선택
        else:
            state_tensor = torch.FloatTensor(state).view(-1).unsqueeze(0)
            q_values = self.network(state_tensor)
            return torch.argmax(q_values).item()  # Q-value가 최대인 행동 선택

    def update_batch(self, memory, batch_size):
        batch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                q_update += self.discount_factor * torch.max(self.network(next_state_tensor)).item()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.network(state_tensor)
            q_values[0][action] = q_update  # 선택한 행동의 Q-value 업데이트

            loss = nn.MSELoss()(self.network(state_tensor), q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update(self, state, action, reward, next_state, done):
        q_update = reward
        if not done:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            q_update += self.discount_factor * torch.max(self.network(next_state_tensor)).item()

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.network(state_tensor)
        q_values[0][action] = q_update  # 선택한 행동의 Q-value 업데이트

        loss = nn.MSELoss()(self.network(state_tensor), q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class A2CLearner(ReinforcementLearner):
    def __init__(self, state_size, action_size, lr, discount_factor, X_train, y_train, X_test, y_test):
        super().__init__(state_size, action_size, lr, discount_factor)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def _build_network(self):
        self.value_network = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Value를 예측하는 네트워크
        )
        self.policy_network = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),  # Policy를 예측하는 네트워크
            nn.Softmax(dim=-1)
        )
        return self.policy_network, self.value_network

    def train(self):
        for epoch in tqdm(range(1000)):
            state = self.X_train[0]
            for t in range(len(self.X_train) - 1):  # 범위를 len(X_train) - 1로 수정
                action, action_prob = self.select_action(state)
                next_state = self.X_train[t + 1]
                reward = self.y_train[t]

                # Advantage를 계산
                advantage = reward + self.discount_factor * self.value_network(torch.FloatTensor(next_state)) - self.value_network(torch.FloatTensor(state))

                # Policy와 Value Network를 각각 업데이트
                self.update_policy(state, action, advantage)
                self.update_value(state, reward, next_state)

                state = next_state

    def update_policy(self, state, action, advantage):
        self.optimizer.zero_grad()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        loss = -torch.log(action_dist.probs[0, action]) * advantage  # 정책 그래디언트
        loss.backward()
        self.optimizer.step()

    def update_value(self, state, reward, next_state):
        value_loss = nn.MSELoss()(self.value_network(torch.FloatTensor(state)), torch.FloatTensor([reward]))
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()