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

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def train(self, X_train, y_train):
        memory = collections.deque(maxlen=10000)
        batch_size = 64

        for epoch in tqdm(range(1000)):  # 학습 에폭
            state = X_train[0]  # 초기 상태 설정
            for t in range(len(X_train)):  # 데이터 길이만큼 반복
                action = self.select_action(state)
                next_state = X_train[t+1]
                reward = y_train[t]

                memory.append((state, action, reward, next_state, False))  # 메모리에 추가
                if len(memory) > batch_size:
                    self.update_batch(memory, batch_size)

                state = next_state  # 상태 업데이트

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)  # 랜덤 행동 선택
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.network(state_tensor)
            return torch.argmax(q_values).item()  # Q-value가 최대인 행동 선택

    def update_batch(self, memory, batch_size):
        batch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update = reward + self.discount_factor * torch.max(self.network(torch.FloatTensor(next_state)))

            q_values = self.network(torch.FloatTensor(state))
            q_values[action] = q_update

            loss = nn.MSELoss()(self.network(torch.FloatTensor(state)), q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


#A2C 클래스
class A2CLearner(ReinforcementLearner):

    #value 와 policy 네트워크 설정
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

    def train(self, X_train, y_train):
        for epoch in tqdm(range(1000)):
            state = X_train[0]
            for t in range(len(X_train)):
                action, action_prob = self.select_action(state)
                next_state = X_train[t+1]
                reward = y_train[t]

                # Advantage를 계산
                advantage = reward + self.discount_factor * self.value_network(torch.FloatTensor(next_state)) - self.value_network(torch.FloatTensor(state))

                # Policy와 Value Network를 각각 업데이트
                self.update_policy(state, action, advantage)
                self.update_value(state, reward, next_state)

                state = next_state

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_probs[0, action.item()]

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