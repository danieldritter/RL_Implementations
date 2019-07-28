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
from A2C import A2C

def __main__():

    # Parses command line arguments
    parser = argparse.ArgumentParser(description="Parses command line arguments for DQN")
    parser.add_argument('--update_type',default="TD", help="Type of update to use, must be either 'TD' or 'Monte_Carlo'")
    parser.add_argument('--render', help="renders game, but slows training significantly", action="store_true")
    parser.add_argument('--gamma', type=float, help="gamma value to use in reward discounting, float in [0,1)", default=.99)
    parser.add_argument('--learning_rate', type=float, help="Learning rate to use in updating network parameters", default=.001)
    parser.add_argument('--num_episodes', type=int, help="Number of episodes to run the agent", default=5000)
    parser.add_argument('--plot', help="Plots average reward every 100 timesteps after training agent", action="store_true")
    parser.add_argument('--num_workers', type=int, help="Number of child processes to spawn to increase speed of learning", default=1)
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
