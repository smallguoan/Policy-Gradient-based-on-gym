import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

class PolicyGradient:
    def __init__(self,learning_rate,n_actions,n_features,reward_decay,output_graph=False):
        self.learning_rate=learning_rate
        self.n_actions=n_actions
        self.n_features=n_features
        self.reward_decay=reward_decay

        self.ep_obs,self.ep_as,self.ep_rs=[],[],[]
        self._build_network()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self):
        with tf.name_scope('inputs'):
            self.tf_obs=tf.placeholder(tf.float32,shape=[None,210,160,3],name='observations')
            self.tf_acts=tf.placeholder(tf.int32,shape=[None,],name='actions_num')
            self.tf_vt=tf.placeholder(tf.float32,shape=[None,],name='action_values')

        with tf.name_scope('conv_layers'):
            conv1 = tf.layers.conv2d(self.tf_obs, 32, [8, 8], strides=(4, 4), padding='SAME', activation=tf.nn.relu,
                                     kernel_regularizer=layers.l2_regularizer(0.1), name='conv1')
            conv2 = tf.layers.conv2d(conv1, 64, [4, 4], strides=(2, 2), padding='SAME', activation=tf.nn.relu,
                                     kernel_regularizer=layers.l2_regularizer(0.1), name='conv2')
            conv3 = tf.layers.conv2d(conv2, 64, [3, 3], strides=(1, 1), padding='SAME', activation=tf.nn.relu,
                                     kernel_regularizer=layers.l2_regularizer(0.1), name='conv3')

        with tf.name_scope('fc_layers'):
            fc1_input = tf.reshape(conv3, [-1, 27 * 20 * 64])
            fc1 = tf.layers.dense(fc1_input, 512, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), name='fc1')
            output = tf.layers.dense(fc1, self.n_actions, activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), name='output')

        self.prob=tf.nn.softmax(output,name='prob')

        with tf.name_scope('loss'):
            cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=self.tf_acts)
            loss=tf.reduce_mean(cross_entropy*self.tf_vt)

        with tf.name_scope('train'):
            self.train_step=tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def choose_action(self,observation):
        act_vec=self.sess.run(self.prob,feed_dict={self.tf_obs:observation[np.newaxis,:]})
        action=np.random.choice(range(act_vec.shape[1]),p=act_vec.ravel())
        #action=np.argmax(act_vec[0])
        #print(action)
        return action
    def store_transition(self,s,a,r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        #print(np.shape(self.ep_obs))
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        self.sess.run(self.train_step,
                      feed_dict={self.tf_obs: self.ep_obs, self.tf_acts: np.array(self.ep_as),self.tf_vt: discounted_ep_rs_norm})
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.reward_decay + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        #discounted_ep_rs=discounted_ep_rs/sum(discounted_ep_rs)
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)+1
        return discounted_ep_rs
