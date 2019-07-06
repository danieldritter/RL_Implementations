import gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

class DQN(nn.Module):

    def __init__(self,output_size):
        super(DQN,self).__init__()

        self.fc1 = nn.Linear(4,256)
        self.fc2 = nn.Linear(256,300)
        self.fc3 = nn.Linear(300,128)
        self.fc4 = nn.Linear(128,output_size)

    def forward(self,input):
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out




class MemoryBuffer:

    def __init__(self,buffer_length,gamma,loss,optimizer,act_net,target_net,batch_size):
        self.buffer_length = buffer_length
        self.gamma = gamma
        self.loss = loss
        self.optimizer = optimizer
        self.replay_buffer = []
        self.act_net = act_net
        self.target_net = target_net
        self.trajectory = []
        self.batch_size = batch_size

    def experience_replay(self):
        for i in range(self.batch_size):
            self.optimizer.zero_grad()
            s_1,a,r,s_2,term = random.choice(self.replay_buffer)
            if not term:
                loss_val = self.loss(r+self.gamma*(torch.max(self.target_net(torch.from_numpy(s_2).float()))),self.act_net(torch.from_numpy(s_1).float())[a])
                loss_val.backward()
                self.optimizer.step()
            else:
                loss_val = self.loss(torch.tensor(r),self.act_net(torch.from_numpy(s_1).float())[a])
                loss_val.backward()
                self.optimizer.step()

    def insert_episode(self,s_1,a,r,s_2,term):
        self.replay_buffer.append((s_1,a,r,s_2,term))
        if len(self.replay_buffer) > self.buffer_length:
            self.replay_buffer.pop(0)

    def monte_carlo_update(self,episode,total_return):
        cumulative_reward = 0
        self.optimizer.zero_grad()
        for s_1,a,r,s_2,term in episode:
            cumulative_reward += r
            loss_val = self.loss(torch.tensor(r+self.gamma*(total_return-cumulative_reward)),self.act_net(torch.from_numpy(s_1).float())[a])
            loss_val.backward()
        self.optimizer.step()

def __main__():
    """
    Simple DQN Implementation for OpenAI Gym Cartpole-v0
    """
    env = gym.make("CartPole-v0")
    # Output size of two for left or right action
    act_net = DQN(2)
    target_net = DQN(2)
    loss = nn.MSELoss()
    optimizer = optim.RMSprop(act_net.parameters())
    gamma = .99
    # Number of iterations before updating target network
    update_iters = 1000
    memory_buffer = MemoryBuffer(100,gamma,loss,optimizer,act_net,target_net,10)
    # Tracks number of iterations for updating target network
    target_counter = 0
    rewards = []
    for i in range(1,1000):
        obs = env.reset()
        episode = []
        epsilon = 100/i
        done = False
        sum_rewards = 0
        for j in range(1000):
            target_counter += 1
            env.render()
            action = torch.argmax(act_net(torch.from_numpy(obs).float())).numpy()
            if random.random() < epsilon:
                action = env.action_space.sample()
            prev = obs
            obs, reward, done, info = env.step(action)
            sum_rewards += reward
            episode.append((prev,action,reward,obs,done))

            # if done:
            #     memory_buffer.insert_episode(prev,action,-reward,obs,done)
            #     memory_buffer.experience_replay()
            #     break
            # else:
            #     # Experience Replay Update
            #     memory_buffer.insert_episode(prev,action,reward,obs,done)
            #     memory_buffer.experience_replay()

            if target_counter == update_iters:
                target_net.load_state_dict(act_net.state_dict())
                target_counter = 0

            if done:
                print("Episode {} finished after {} timesteps".format(i,j+1))
                print(sum_rewards)
                rewards.append(sum_rewards)
                memory_buffer.monte_carlo_update(episode,sum_rewards)
                break
    env.close()
    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    __main__()
