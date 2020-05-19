"""
Implementation of DQN on openai gym's cartpole environment
"""

import gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import torch.nn.utils as utils

class DQN(nn.Module):
    """
    Network to approximate Q function
    """
    def __init__(self,output_size):
        super(DQN,self).__init__()

        self.fc1 = nn.Linear(4,64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,96)
        self.fc4 = nn.Linear(96,output_size)

    def forward(self,input):
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out




class MemoryBuffer:
    """
    Class representing experience replay buffer to sample episodes from
    """
    def __init__(self,buffer_length,gamma,loss,optimizer,act_net,target_net,batch_size):
        """
        ::Params::
            buffer_length (int): maximum size of experience replay buffer
            gamma (float in [0,1)): gamma discount factor to use when calculating discounted rewards
            loss (torch.nn loss function): Loss function to use in optimizing Q Network
            optimizer (torch.optim optimizer): The optimizer to call to backpropagate gradients through the Q Network
            act_net (DQN): The actor network that approximates the Q Value
            target_net (DQN): The target network with temporarily frozen parameters that the actor network is evaluated against
            batch_size (int): The number of steps to sample in one update
        ::Output::
            None
        """
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
        """
        This method samples a random set of episodes of size batch size
        from the replay_buffer and optimizes the Q-Network using them.
        """
        self.optimizer.zero_grad()
        for i in range(self.batch_size):
            s_1,a,r,s_2,term = random.choice(self.replay_buffer)
            q_values = self.act_net(torch.from_numpy(s_1).float())
            # Remove bootstrapping term if final step(since reward after final step shold be 0)
            if term:
                loss_val = self.loss(torch.tensor(r),q_values[a])
            else:
                # Take max over target network output to approximate V(s')
                loss_val = self.loss(r+self.gamma*(torch.max(self.target_net(torch.from_numpy(s_2).float()))),q_values[a])
            loss_val.backward()
        self.optimizer.step()

    def insert_episode(self,s_1,a,r,s_2,term):
        """
        Stores a tuple of (s_1,a,r,s_2,term) in replay buffer to sample from
        ::Params::
            s_1 (env observation): initial state from gym environment
            a (int): action taken from state s_1
            r (float): reward received after taking action a in state s_1
            s_2 (env observation): State entered after taking action a in state s_1
            term (boolean): Whether or not the episode ended after this step
        ::Output::
            None
        """
        self.replay_buffer.append((s_1,a,r,s_2,term))
        # Limits size of replay to buffer_length
        if len(self.replay_buffer) > self.buffer_length:
            self.replay_buffer.pop(0)

    def monte_carlo_update(self,episode,total_return):
        """
        Rather than using the replay buffer, there is also a command line flag
        to update using monte carlo returns instead of td bootstrapping from a
        replay buffer. If that flag is passed, then this function will be called
        instead of insert_episode or experience_replay. It calculates the loss
        using discounted rewards over a complete episode, and then optimizes
        the network with respect to that. Be aware this is very high variance
        and sometimes will diverge with the default hyperparameters.
        ::Params::
            episode (list): A list of (state, action, reward, state_prime, done) tuples representing steps of the episode.
                See insert_episode docstring for specifics on parts of the tuple
            total_return (float): This is total amount of reward received during the episode.
        ::Output::
            None
        """
        cumulative_reward = 0
        self.optimizer.zero_grad()
        for s_1,a,r,s_2,term in episode:
            cumulative_reward += r
            # Remove cumulative reward component on final step
            if term:
                # Give zero reward on last step to disincentivize that state
                r = 0
                loss_val = self.loss(torch.tensor(r),self.act_net(torch.from_numpy(s_1).float())[a])
            else:
                loss_val = self.loss(torch.tensor(r+self.gamma*(total_return-cumulative_reward)),self.act_net(torch.from_numpy(s_1).float())[a])
            loss_val.backward()
        self.optimizer.step()

def __main__():
    """
    Simple DQN Implementation for OpenAI Gym Cartpole-v0

    Usage: python DQN.py + assorted flags below
    """
    # Parses command line arguments
    parser = argparse.ArgumentParser(description="Parses command line arguments for DQN")
    parser.add_argument('--update_type',default="TD", help="Type of update to use, must be either 'TD' or 'Monte_Carlo'")
    parser.add_argument('--render', help="renders game, but slows training significantly", action="store_true")
    parser.add_argument('--buffer_size', type=int, help="Size of experience replay buffer", default=1000)
    parser.add_argument('--gamma', type=float, help="gamma value to use in reward discounting, float in [0,1)", default=.99)
    parser.add_argument('--epsilon', type=float, help="Epsilon value to use in determining random vs. greedy actions", default=1.0)
    parser.add_argument('--epsilon_decay', type=float, help="Value to decay epsilon by after each episode(epsilon *= epsilon_decay). In range [0,1)", default=.9995)
    parser.add_argument('--learning_rate', type=float, help="Learning rate to use in updating network parameters", default=.001)
    parser.add_argument('--target_network_lag', type=int, help="Number of episodes to wait before updating target network parameters", default=20)
    parser.add_argument('--batch_size', type=int, help="Size of batches to sample from experience replay buffer(Ignored for monte carlo updating)", default=10)
    parser.add_argument('--num_episodes', type=int, help="Number of episodes to run the agent", default=2000)
    parser.add_argument('--plot', help="Plots average reward every 100 timesteps after training agent", action="store_true")
    args = parser.parse_args()

    # Checks that update type is supported
    if args.update_type not in ["TD","Monte_Carlo"]:
        print("Update Type must be one of ['TD','Monte_Carlo']")

    # Stores some of the arguments in local variables
    epsilon = args.epsilon
    env = gym.make("CartPole-v1")

    # Output size of two for left or right action
    act_net = DQN(env.action_space.n)
    target_net = DQN(env.action_space.n)
    # Create loss and optimizer
    loss = nn.MSELoss()
    optimizer = optim.Adam(act_net.parameters(), lr=args.learning_rate)

    # Number of episodes before updating target network
    update_iters = args.target_network_lag
    # Create buffer
    memory_buffer = MemoryBuffer(args.buffer_size, args.gamma, loss, optimizer, act_net, target_net, args.batch_size)
    # Tracks number of iterations for updating target network
    target_counter = 0
    # Stores total reward at end of every episode
    rewards = []
    for i in range(args.num_episodes):
        target_counter += 1
        obs = env.reset()
        episode = []
        # Decays epsilon to reduce randomness over time
        epsilon *= args.epsilon_decay
        done = False
        sum_rewards = 0
        for j in range(1000):
            if args.render:
                env.render()
            with torch.no_grad():
                action = torch.argmax(act_net(torch.from_numpy(obs).float())).numpy()
            if random.random() < epsilon:
                action = env.action_space.sample()
            prev = obs
            obs, reward, done, info = env.step(action)
            sum_rewards += reward
            episode.append((prev,action,reward,obs,done))

            if args.update_type == "TD":
                memory_buffer.insert_episode(prev,action,reward,obs,done)

            if done:
                print("Episode {} finished after {} timesteps".format(i,j+1))
                print(sum_rewards)
                # Stores reward to graph if plotting is turned on
                if args.plot:
                    rewards.append(sum_rewards)
                if args.update_type == "Monte_Carlo":
                    memory_buffer.monte_carlo_update(episode,sum_rewards)
                else:
                    memory_buffer.experience_replay()
                # Updates target network
                if target_counter == update_iters:
                    target_net.load_state_dict(act_net.state_dict())
                    target_counter = 0
                break
    env.close()

    # Plots average rewards
    if args.plot:
        avg_rewards = []
        for i in range(0,len(rewards),100):
            avg_rewards.append(np.mean(rewards[i:i+100]))
        plt.plot(avg_rewards)
        plt.show()


if __name__ == "__main__":
    __main__()
