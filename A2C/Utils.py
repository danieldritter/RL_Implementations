

class MultiEnv():
    """
    Class to run multiple environments in parallel and return all of their information
    """
    def __init__(self, num_envs):
        self.envs = [gym.make("CartPole-v1") for i in range(num_envs)]


    def get_n_step_observations(self, n):
        """
        This method generates a series of n step observations in parallel
        to train the A2C agent. A value of 0 will generate complete episodes(Monte Carlo)
        while 1 is equivalent to TD updates
        """

    def _get_observation(self, env, n):
        states = []
        actions = []
        rewards = []
        for i in range(n):



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
