import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.CNN import CNNActionValue

class DQN:
    def __init__(
            self,
            state_dim,                    #입력 이미지 상태의 크기
            action_dim,                   #agent가 선택할 수 있는 행동의 개수
            lr=0.001,                     #학습률
            epsilon=1.0,                  #epsilon-greedy 1.0에서 0.1으로 감소
            epsilon_min=0.1,              
            gamma=0.9,                    #discount factor
            batch_size=32,                #sample 묶는 개수
            warmup_steps=5000,            #훈련 시작 전 랜덤으로 행동하는 개수
            buffer_size=int(1e5),         #buffer의 최대 크기
            target_update_interval=10000, #target network 업데이트 주기
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = CNNActionValue(state_dim[0], action_dim)               #현재 업데이트하는 네트워크
        self.target_network = CNNActionValue(state_dim[0], action_dim)        #target 네트워크
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)
        # self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #for mac

        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e5  #epsilon을 1e6 step 동안 감소

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):  #epsilon 확률로 행동 선택하고 이를 warmup-step 동안 지속
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device) #현재 상태 x를 tensor로 변환(numpy배열 -> tensor, 차원 추가, GPU로 이동)
            q = self.network(x) #전처리된 상태 x를 네트워크에 넣어 q값을 얻음
            a = torch.argmax(q).item() #q값이 가장 큰 행동을 선택
        return a

    #double DQN
    def learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size)) #buffer에서 batch_size만큼 샘플링

        next_q = self.network(s_prime).detach() #.detach()는 gradient를 계산하지 않도록 함
        next_action = next_q.argmax(dim=1, keepdim=True)
        
        next_q_target = self.target_network(s_prime).detach()
        next_q_value = next_q_target.gather(1, next_action) #target network에서 다음 상태에 대한 q값을 가져옴
        
        td_target = r + (1. - terminated) * self.gamma * next_q_value
        loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)
        self.optimizer.zero_grad()  #역전파로 네트워크 업데이트
        loss.backward() #pytorch에서 역전파는 loss.backward()로 수행
        self.optimizer.step()

        result = {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }
        return result

    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition) #(s,a,r,s',terminated) tuple을 buffer에 저장

        if self.total_steps > self.warmup_steps: #warmup step이 지나면 학습 시작
            result = self.learn()

        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict()) #target network 업데이트
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        #self.epsilon -= self.epsilon_decay
        return result


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]),
        )