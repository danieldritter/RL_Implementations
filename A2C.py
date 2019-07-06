import tensorflow as tf
import gym
import numpy as np
import time

class Actor:

    def __init__(self,session,action_space_sz,observation_space_sz):
        self.session = session
        self.action_space_sz = action_space_sz
        self.observation_space_sz = observation_space_sz
        self.a_state_input = tf.placeholder(tf.float32,[None,observation_space_sz])
        self.a_actions = tf.placeholder(tf.int32,[None])
        self.a_rewards = tf.placeholder(tf.float32,[None])
        self.a_action_dist = self.action_probs()
        self.critic_preds = tf.placeholder(tf.float32,[None])
        self.a_loss_value = self.loss()
        self.a_train = self.optimizer()

    def loss(self):
        chosen_actions = tf.gather(tf.reshape(self.a_action_dist,[-1]),self.a_actions)
        return tf.reduce_mean(-tf.log(chosen_actions)*(self.a_rewards-self.critic_preds))

    def action_probs(self):
        WA1 = tf.Variable(tf.random_normal([self.observation_space_sz,32],stddev=.1))
        BA1 = tf.Variable(tf.random_normal([32],stddev=.1))
        WA2 = tf.Variable(tf.random_normal([32,64],stddev=.1))
        BA2 = tf.Variable(tf.random_normal([64],stddev=.1))
        WA3 = tf.Variable(tf.random_normal([64,self.action_space_sz],stddev=.1))
        BA3 = tf.Variable(tf.random_normal([self.action_space_sz],stddev=.1))
        Aout1 = tf.nn.relu(tf.matmul(self.a_state_input,WA1) + BA1)
        Aout2 = tf.nn.relu(tf.matmul(Aout1,WA2) + BA2)
        Aout3 = tf.nn.softmax(tf.matmul(Aout2,WA3) + BA3)
        return Aout3

    def get_action(self,obs):
        action = self.session.run(tf.reshape(self.a_action_dist,[self.action_space_sz]),feed_dict={self.a_state_input:[obs]})
        action = np.random.choice(np.arange(self.action_space_sz),1,p=action)
        return action[0]

    def optimizer(self):
        return tf.train.AdamOptimizer(.001).minimize(self.a_loss_value)

    def train(self,states,actions,rewards,pred_values):
        self.session.run(self.a_train,feed_dict={self.a_state_input:states,self.a_actions:actions,self.a_rewards:rewards,self.critic_preds:pred_values})

class Critic:

    def __init__(self,session,action_space_sz,observation_space_sz):
        self.session = session
        self.action_space_sz = action_space_sz
        self.observation_space_sz = observation_space_sz
        self.c_gamma = .99
        self.c_state_input = tf.placeholder(tf.float32,[None,observation_space_sz])
        self.c_successor_states = tf.placeholder(tf.float32,[None,observation_space_sz])
        self.c_actions = tf.placeholder(tf.float32,[None])
        self.c_rewards = tf.placeholder(tf.float32,[None])
        self.c_pred_values = self.value_prediction(self.c_state_input)
        self.c_successor_values = self.value_prediction(self.c_successor_states)
        self.c_loss_value = self.loss()
        self.c_train = self.optimizer()

    def loss(self):
        return tf.reduce_mean(tf.square(-self.c_pred_values+(self.c_rewards+self.c_gamma*self.c_successor_values)))

    def value_prediction(self,states):
        WC1 = tf.Variable(tf.random_normal([self.observation_space_sz,32],stddev=.1))
        BC1 = tf.Variable(tf.random_normal([32],stddev=.1))
        WC2 = tf.Variable(tf.random_normal([32,64],stddev=.1))
        BC2 = tf.Variable(tf.random_normal([64],stddev=.1))
        WC3 = tf.Variable(tf.random_normal([64,1],stddev=.1))
        BC3 = tf.Variable(tf.random_normal([1],stddev=.1))
        Cout1 = tf.nn.relu(tf.matmul(states,WC1) + BC1)
        Cout2 = tf.nn.relu(tf.matmul(Cout1,WC2) + BC2)
        Cout3 = tf.matmul(Cout2,WC3) + BC3
        return Cout3

    def optimizer(self):
        return tf.train.AdamOptimizer(.001).minimize(self.c_loss_value)


    def train(self,states,actions,rewards,states_prime):
        self.session.run(self.c_train,feed_dict={self.c_state_input:states,self.c_actions:actions,self.c_rewards:rewards,self.c_successor_states:states_prime})


def __main__():
    replay_buffer = []
    replay_buffer_length = 20
    minibatch_sz = 10
    env = gym.make("CartPole-v0")
    sess = tf.Session()
    critic = Critic(sess,env.action_space.n,4)
    actor = Actor(sess,env.action_space.n,4)
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        obs = env.reset()
        done = False
        rewards = 0
        while not done:
            env.render()
            action = actor.get_action(obs)
            prev = obs
            obs, reward, done, _ = env.step(action)
            rewards += reward
            critic.train([prev],[action],[rewards],[obs])
            pred_values = sess.run(tf.reshape(critic.c_pred_values,[1]),feed_dict={critic.c_state_input:[obs]})
            print(pred_values)
            actor.train([prev],[action],[rewards],pred_values)


if __name__ == "__main__":
    __main__()
