import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import gym
import numpy as np
import time
import random
import argparse

class A2C(nn.Module):
    """
    Network to predict both actor policy and critic values
    """
    def __init__(self, state_size, n_actions):
        super(A2C, self).__init__()
        # Actor Layers
        self.act1 = nn.Linear(state_size, 32)
        self.act2 = nn.Linear(32, 64)
        self.act3 = nn.Linear(64, n_actions)
        # Critic Layers
        self.crit1 = nn.Linear(state_size, 32)
        self.crit2 = nn.Linear(32, 64)
        self.crit3 = nn.Linear(64, 1)


    def forward(self, input):
        """
        Method to pass an input through both the actor and critic networks
        ::Params::
            input (Tensor): Input tensor to the networks, representing the current state
        ::Output::
            Policy (Tensor of shape (n_actions)): Probability distribution over actions
            Value: (Tensor of shape (1)): Predicted value of state
        """
        # Actor prediction
        policy = F.relu(self.act1(input))
        policy = F.relu(self.act2(policy))
        policy = F.softmax(self.act3(policy))

        # Critic prediction
        value = F.relu(self.crit1(input))
        value = F.relu(self.crit2(value))
        value = self.crit3(value)
        return policy, value

def generate_trajectory(env, network, render):
    """
    This method generates a trajectory of (state, action, reward) triplets and
    then returns them in separate lists along with the total reward of the episode
    ::Params::
        env (gym environment): The openai gym environment to call step() on
        network (nn.Module): The A2C network to use to determine the policy
        render (boolean): Flag to determine whether to render the environment or not
    """
    states, actions, rewards = [], [] ,[]

    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        if render:
            env.render()
        policy, value = network(torch.from_numpy(obs).float())
        action = np.random.choice(env.action_space.n,p=policy.detach().numpy())
        states.append(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        total_reward += reward
    return states, actions, rewards, total_reward

def discount_rewards(rewards, gamma):
    """
    This method takes in a list of rewards for one trajectory and generates
    a list of the discounted sum of rewards to use in updating the network.
    ::Params::
        rewards (list): List of rewards from environment for one episode
        gamma (float in [0,1)): Discount factor to use in calculating discounted rewards
    ::Output::
        discounted_rewards (list): List of discounted rewards, where discounted_rewards[i] = rewards[i] + sum(gamma*rewards[i:])
    """
    prev = 0
    discounted_rewards = np.copy(rewards)
    for i in range(1, len(rewards) + 1):
        discounted_rewards[-i] += gamma*prev
        prev = discounted_rewards[-i]
    return discounted_rewards

def __main__():

    # Parses command line arguments
    parser = argparse.ArgumentParser(description="Parses command line arguments for DQN")
    parser.add_argument('--update_type',default="TD", help="Type of update to use, must be either 'TD' or 'Monte_Carlo'")
    parser.add_argument('--render', help="renders game, but slows training significantly", action="store_true")
    parser.add_argument('--gamma', type=float, help="gamma value to use in reward discounting, float in [0,1)", default=.99)
    parser.add_argument('--learning_rate', type=float, help="Learning rate to use in updating network parameters", default=.001)
    parser.add_argument('--num_episodes', type=int, help="Number of episodes to run the agent", default=5000)
    parser.add_argument('--plot', help="Plots average reward every 100 timesteps after training agent", action="store_true")
    args = parser.parse_args()

    env = gym.make("CartPole-v1")
    network = A2C(len(env.observation_space.high),env.action_space.n)
    optimizer = optim.Adam(network.parameters(),lr=args.learning_rate)

    for i in range(args.num_episodes):
        if args.update_type == "Monte_Carlo":
            states, actions, rewards, total_reward = generate_trajectory(env, network, args.render)
            print(total_reward)
            discounted_rewards = discount_rewards(rewards, args.gamma)
            optimizer.zero_grad()
            for state, action, reward in zip(states, actions, discounted_rewards):
                policy, value = network(torch.from_numpy(state).float())
                critic_loss_val = (reward - value)**2
                actor_loss_val = -torch.log(policy[action])*(reward - value.detach())
                loss = .5*critic_loss_val + actor_loss_val
                loss.backward()
            optimizer.step()
        elif args.update_type == "TD":
            done = False
            obs = env.reset()
            total_reward = 0.0
            while not done:
                # Samples action from current policy
                policy, value = network(torch.from_numpy(obs).float())
                action = np.random.choice(env.action_space.n,p=policy.detach().numpy())

                # Renders if flag is passed
                if args.render:
                    env.render()

                s_prime, reward, done, _ = env.step(action)
                total_reward += reward
                # Update networks
                optimizer.zero_grad()
                s_prime_policy, s_prime_val = network(torch.from_numpy(s_prime).float())
                # Don't bootstrap on terminal state
                if not done:
                    advantage = reward + args.gamma*s_prime_val - value
                else:
                    advantage = reward - value
                critic_loss_val = advantage**2
                actor_loss_val = -torch.log(policy[action])*advantage.detach()
                loss = .5*critic_loss_val + actor_loss_val
                loss.backward()
                optimizer.step()
                obs = s_prime
            print(total_reward)
    env.close()


if __name__ == "__main__":
    __main__()
