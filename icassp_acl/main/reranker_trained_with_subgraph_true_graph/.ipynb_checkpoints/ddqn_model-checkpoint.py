import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import re
import numpy as np
import random
from collections import deque
from torch.utils.checkpoint import checkpoint
import os

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, actions, next_actions, reward, done):
        experience = (state, action, actions, next_actions, np.array([reward]), done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        actions_batch = []
        next_actions_batch = []
        reward_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, actions, next_actions, reward, done = experience
            state_batch.append(state)
            action_batch.append(action)
            actions_batch.append(actions)
            next_actions_batch.append(next_actions)
            reward_batch.append(reward)
            done_batch.append(done)

        return (state_batch, action_batch, actions_batch, next_actions_batch, reward_batch, done_batch)

    def __len__(self):
        return len(self.buffer)

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        whole_output = self.encoder(input_ids, attention_mask).logits
        return whole_output


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.roberta_pretrain = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=1)
        self.roberta = EncoderWrapper(self.roberta_pretrain)
        self.max_seq_length = 512
        self.query_length = 195
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.roberta.to(self.device)
    def forward(self, query, passages):
        if isinstance(query, str):
            query_ids = self.tokenizer([query], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
                        :self.query_length]
            query = self.tokenizer.decode(query_ids)
            texts_b = passages
            texts_a = [query] * len(texts_b)
            inputs = self.tokenizer(
                texts_a, texts_b,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=self.max_seq_length,
                padding='longest',
                truncation=True)
            inputs = {n: t.to(self.device) for n, t in inputs.items()}
            dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True).to(self.device)
            qvals = checkpoint(self.roberta, inputs['input_ids'], inputs['attention_mask'], dummy_tensor)
        else:
            all_batch_result = []
            for every_query, every_passages in zip(query, passages):
                query_ids = self.tokenizer([every_query], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
                            :self.query_length]
                query = self.tokenizer.decode(query_ids)
                texts_b = every_passages
                texts_a = [query] * len(texts_b)
                inputs = self.tokenizer(
                    texts_a, texts_b,
                    add_special_tokens=True,
                    return_tensors='pt',
                    max_length=self.max_seq_length,
                    padding='longest',
                    truncation=True)
                inputs = {n: t.to(self.device) for n, t in inputs.items()}
                dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True).to(self.device)
                qvals = checkpoint(self.roberta, inputs['input_ids'], inputs['attention_mask'], dummy_tensor)
                all_batch_result.append(qvals)
            qvals = all_batch_result
        return qvals


class DQNAgent:

    def __init__(self, learning_rate=3e-6, gamma=0.99, tau=0.01, buffer_size=10000):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.env = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.select_mem = []
        self.labels = []
        self.model1 = DQN().to(self.device)
        self.model2 = DQN().to(self.device)
        self.level = -1  # 0表示顶层
        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr = self.learning_rate)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr = self.learning_rate)
        self.count = 0
        self.save_addr = '/mnt/huihuan/huihuan/acl_code/save_test'
        
    def get_action(self, state, eps=0.20):
        eps = eps * 0.25 + eps * 0.75 *(1- self.count/600)
        action_space = list(self.env.keys())
        select_hitory = ' // '.join(self.select_mem)
        if len(self.select_mem):
            action_space = [select_hitory + ' // ' + x for x in action_space]
        else:
            action_space = [x for x in action_space]
        qvals = self.model1.forward(state, action_space)
        action = action_space[np.argmax(qvals.cpu().detach().numpy())]
        if (np.random.randn() < eps):
            action = action_space[random.randint(0, len(action_space) - 1)]
            return action, action_space
        return action, action_space

    def compute_loss(self, batch):
        states, actions, actions_list, next_actions_list, rewards, dones = batch
        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        # next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        # resize tensors
        dones = dones.view(dones.size(0), 1)
        action_index = []
        for every_action_list_index in range(len(actions_list)):
            now_action_list = actions_list[every_action_list_index]
            now_action = actions[every_action_list_index]
            action_index.append(now_action_list.index(now_action))
        #action_index = torch.FloatTensor(action_index).to(self.device)
        #action_index = action_index.view(action_index.size(0), 1)
        # compute loss
        curr_Q1_all = self.model1.forward(states, actions_list)
        curr_Q2_all = self.model2.forward(states, actions_list)
        curr_Q1 = []
        curr_Q2 = []
        for every_action_index in range(len(action_index)):
            now_action_index = action_index[every_action_index]
            curr_Q1.append(curr_Q1_all[every_action_index][now_action_index].unsqueeze(0))
            curr_Q2.append(curr_Q2_all[every_action_index][now_action_index].unsqueeze(0))
        curr_Q1 = torch.cat(curr_Q1)
        curr_Q2 = torch.cat(curr_Q2)
        next_Q1 = self.model1.forward(states, next_actions_list)
        next_Q2 = self.model2.forward(states, next_actions_list)
        next_Q = []
        for every_action_index in range(len(next_Q1)):
            now_Q1_max = torch.max(next_Q1[every_action_index])
            now_Q2_max = torch.max(next_Q2[every_action_index])
            next_Q.append(torch.min(now_Q1_max, now_Q2_max).unsqueeze(0))
        next_Q = torch.cat(next_Q)
        next_Q = next_Q.view(next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * next_Q

        loss1 = F.mse_loss(curr_Q1, expected_Q.detach())
        loss2 = F.mse_loss(curr_Q2, expected_Q.detach())

        return loss1, loss2

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss1, loss2 = self.compute_loss(batch)

        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
        self.count += 1
        if self.count % 20 == 0:
            print('loss1: ', loss1)
            print('loss2: ', loss2)
            
    def reset(self, passages, labels_id):
        env = {}
        self.level = -1
        for every_passage in passages:
            now_env = env
            now_passage = every_passage['text']
            candidate_dict = re.split('<1> |<2> |<3> ', now_passage)[1:]
            if every_passage['pid'] in labels_id:
                self.labels.append(candidate_dict)
            for every_title in candidate_dict:
                if every_title in now_env:
                    now_env = now_env[every_title]
                else:
                    now_env[every_title] = {}
                    now_env = now_env[every_title]
        self.env = env

    def step(self, action):
        action = action.split(' // ')[-1]
        self.select_mem.append(action)
        self.env = self.env[action]
        self.level += 1
        if len(self.env.keys()) == 0:
            done = 1
            if self.select_mem in self.labels:
                reward = 2
            else:
                reward = -1
            self.select_mem = []
            self.labels = []
        else:
            done = 0
            for every_label in self.labels:
                if every_label[self.level] == action:
                    # 该奖励不该比下面的惩罚还大
                    reward = 0.1
                else:
                    reward = -0.2
        next_action_space = list(self.env.keys())
        select_hitory = ' // '.join(self.select_mem)
        if done:
            next_action_space = ['']
        else:
            next_action_space = [select_hitory + ' // ' + x for x in next_action_space]
        return reward, done, next_action_space
    
    
    def save(self, addr=None):
        if addr is None:
            if not os.path.exists(self.save_addr):
                os.mkdir(self.save_addr)
            torch.save(self.model1.state_dict(), self.save_addr+'/model1.bin')
            torch.save(self.model2.state_dict(), self.save_addr+'/model2.bin')
        else:
            if not os.path.exists(addr):
                os.mkdir(addr)
            torch.save(self.model1.state_dict(), addr+'/model1.bin')
            torch.save(self.model2.state_dict(), addr+'/model2.bin')