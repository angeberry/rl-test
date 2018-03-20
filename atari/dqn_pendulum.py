import tensorflow as tf
import numpy as np
import gym
import random
import os

"""
def qnet(input, act_n):
    initializer = tf.contrib.layers.xavier_initializer()
    x = tf.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu, kernel_initializer=initializer)
    x = tf.layers.conv2d(x, 64, 64, 4, 2, activation_fn=tf.nn.relu, kernel_initializer=initializer)
    x = tf.layres.conv2d(x, 64, 3, 1, activation_fn=tf.nn.relu, kernel_initializer=initializer)

    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dense(x, act_n)

    return x
"""
global_step = tf.Variable(0, name="global_step", trainable=False)

def qnet(inp, act_n):
    # initializer = tf.contrib.layers.xavier_initializer()
    x = tf.layers.dense(inp, 16, activation=tf.nn.relu)
    x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    x = tf.layers.dense(x, 256, activation=tf.nn.relu)
    x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
    x = tf.layers.dense(x, act_n)
    return x


def random_player(act_n):
    return np.expand_dims(random.choice(list(range(act_n))), 0)


def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


class copy_parameters(object):
    def __init__(self, estimator1, estimator2):
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.name)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.name)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def make(self, sess):
        sess.run(self.update_ops)


class Estimator(object):
    def __init__(self, sess, summaries_dir='/home/hzyuan/DRL/DQN', batch_size=32, act_n=10, name='q_estimator'):
        self.sess = sess
        self.summary_writer = None
        self.summaries_dir = summaries_dir
        self.batch_size = batch_size
        self.act_n = act_n
        self.X = tf.placeholder(tf.float32, [None, 3], 'x')
        self.y = tf.placeholder(tf.float32, [None, 1], 'y')
        self.act = tf.placeholder(tf.int32, [None, 1], 'act')
        self.name = name
        self.ep = tf.placeholder(tf.float32, (), 'epsilon')
        self.reward = tf.placeholder(tf.float32, (), 'reward')
        self._build_model()

    def _build_model(self):
        with tf.variable_scope(self.name) as scope:
            self.predictions = qnet(self.X, self.act_n)
            matrix = tf.one_hot(tf.reshape(self.act, [-1]), self.act_n, 0.9, 0.0, dtype=tf.float32)
            print(self.predictions)
            print(matrix)
            matrix = self.predictions * matrix

            self.act_pred = tf.reduce_sum(matrix, 1)

            # gather_indices = tf.range(self.batch_size) * tf.shape(self.predictions)[1] + self.act
            # self.act_pred = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

            self.losses = tf.nn.l2_loss(self.act_pred - tf.reshape(self.y, [-1]))
            self.loss = tf.reduce_mean(self.losses)


            self.optimizer = tf.train.RMSPropOptimizer(0.0025, 0.99, 0.0, 1e-6)
            #self.optimizer = tf.train.AdamOptimizer(0.003)
            self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

            self.init = tf.global_variables_initializer()

            if self.summaries_dir:
                summary_dir = os.path.join(self.summaries_dir, 'summaries_{}'.format(self.name))
                self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

            self.summaries = tf.summary.merge(
                [tf.summary.scalar('loss', self.loss), tf.summary.histogram('loss_hist', self.losses)])
            self.outer_sums = tf.summary.merge([tf.summary.scalar('epsilon', self.ep), tf.summary.scalar('reward', self.reward)])
            self.sess.run(self.init)

    def predict(self, state):
        self.pred = self.sess.run(self.predictions, feed_dict={self.X: state})
        return self.pred

    def update(self, state, action, value):
        _, loss, act_p, summaries = self.sess.run(
            [self.train_op, self.loss, self.act_pred, self.summaries],
            feed_dict={self.X: state, self.act: action, self.y: value})
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step=self.sess.run(tf.train.get_global_step()))
            #print(step)
        return loss


def DQN(env, sess, q_estimator, processer, episodes=3000, discount_factor=0.9, max_memory=10000, train_steps=100,
        batch_size=32, act_n=10, update_steps=500):
    memory = []
    count = 0
    done = False
    copier = copy_parameters(q_estimator, processer)
    runner = make_epsilon_greedy_policy(processer, act_n)
    resu = []
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 100000
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay)

    episode_summary = tf.Summary()
    update_summary = tf.Summary()
    for i in range(episodes):
        done = False
        state = env.reset()
        ttl_r = 0
        while not done:
            epsilon = epsilons[min(count, (epsilon_decay-1))]
            # print(epsilon)
            probs = runner(sess, np.expand_dims(state, 0), epsilon)
            action = np.random.choice(range(act_n), p=np.squeeze(probs))
            action = np.expand_dims(action, 0)
            next_state, reward, done, _ = env.step((action /2.5) - 2)
            ttl_r += reward
            # print('reward:%f'%reward)
            memory.append((state, action, reward, next_state, done))
            if len(memory) > max_memory:
                memory.pop(0)
            if (count % train_steps == 0) & (count > batch_size):
                train_batch = random.sample(memory, batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = list(
                    map(np.array, zip(*train_batch)))
                q = q_estimator.predict(next_state_batch)
                next_q = np.max(q, 1)
                target_batch = discount_factor * next_q + reward_batch
                target_batch = np.expand_dims(target_batch, 1)
                # print(target_batch[0])
                q_estimator.update(state_batch, action_batch, target_batch)

            if (count % update_steps == 0) & (not count == 0):
                copier.make(sess)

            state = next_state
            count += 1
        resu.append(ttl_r)

        update_summary.value.add(simple_value=np.mean(next_q), tag='q_mean')
        update_summary.value.add(simple_value=np.var(next_q), tag='q_var')
        print(ttl_r)
        episode_summary.value.add(simple_value=epsilon, tag='episode/epsilon')
        print(epsilon)
        print(i)
        episode_summary.value.add(simple_value=ttl_r, tag='episode/reward')

        summaries = sess.run(q_estimator.outer_sums, feed_dict = {q_estimator.ep: epsilon, q_estimator.reward:ttl_r})
        q_estimator.summary_writer.add_summary(update_summary, count)
        q_estimator.summary_writer.add_summary(summaries, sess.run(tf.train.get_global_step()))
        q_estimator.summary_writer.flush()
    resu = np.array(resu)
    # episode_summary.value.add(simple_value=resu, tag = 'episode/reward')
    print(epsilon)
    return (np.array(resu))

if __name__ == '__main__':
    sess = tf.Session()
    env = gym.make('Pendulum-v0')
    q_estimator = Estimator(sess)
    processer = Estimator(sess, name='processer')
    test = DQN(env, sess, q_estimator, processer)

"""
import matplotlib.pyplot as plt
state = env.reset()
state = np.expand_dims(state, 0)
done = False
while not done:
    p = processer.predict(state)
    action = np.argmax(p, 1)
    #print(state)
    print(action)
    print(p)
    action = np.expand_dims(action, 0)
    next_state, reward, done, _ = env.step((action*4)-2)
    state = np.transpose(next_state)
    test = env.render(mode='rgb_array')
    plt.imshow(test)
    plt.show()
"""
