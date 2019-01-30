import os
import shutil
import time
import logging
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
import numpy as np

class CriticModel(keras.Model):
    def __init__(self, observation_shape, action_shape, hidden_size=128):
        super(CriticModel, self).__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        
        self.value_dense_1 = layers.Dense(hidden_size, activation='relu')
        self.value_dense_2 = layers.Dense(hidden_size, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        input_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        y = self.value_dense_1(input_tensor)
        y = self.value_dense_2(y)
        values = self.values(y)
        return values
    
class ActorModel(keras.Model):
    def __init__(self, observation_shape, action_shape, hidden_size=128):
        super(ActorModel, self).__init__()
        self.observation_size = observation_shape
        self.action_size = action_shape
        
        self.policy_dense_1 = layers.Dense(hidden_size, activation='relu')
        self.policy_dense_2 = layers.Dense(hidden_size, activation='relu')
        self.policy_logits = layers.Dense(action_shape)

    def call(self, inputs):
        input_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = self.policy_dense_1(input_tensor)
        x = self.policy_dense_2(x)
        logits = self.policy_logits(x)
        return logits
    
class ActorCriticModel(keras.Model):
    def __init__(self, observation_shape, action_shape, hidden_size=128):
        super(ActorCriticModel, self).__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.actor = ActorModel(observation_shape, action_shape, hidden_size)
        self.critic = CriticModel(observation_shape, action_shape, hidden_size)
        
    def call(self, inputs):
        return self.actor(inputs), self.critic(inputs)
        
class Agent:
    def __init__(self, env, gamma=0.99, ent_coef=0.001, lr=0.001, critic_coef=0.5,
                hidden_size=128, epochs=8, batch_size=512, *args, **kwargs):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # Set hyperparameters
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.lr = lr
        self.critic_coef = critic_coef
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = ActorCriticModel(self.observation_space.shape, self.action_space.n, self.hidden_size)
    
    def loss(self, observations, actions, discounted_rewards, normalize_discounted_rewards=True, *args, **kargs):
        logits, values = self.model(tf.convert_to_tensor(observations, dtype=tf.float32))
        values = tf.squeeze(values)

        if normalize_discounted_rewards:
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

        discounted_rewards = tf.convert_to_tensor(np.array(discounted_rewards), dtype=tf.float32)
        advantages =  tf.stop_gradient(discounted_rewards) - values

        critic_loss = advantages ** 2

        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

        actor_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
        actor_loss *= tf.stop_gradient(advantages)
        actor_loss -= self.ent_coef * entropy

        total_loss = tf.reduce_mean(self.critic_coef * critic_loss + actor_loss)

        return total_loss

    def learn(self, observations, actions, discounted_rewards, *args, **kargs):
        opt = tf.train.AdamOptimizer(self.lr, use_locking=True)
        with tf.GradientTape() as tape:
            total_loss = self.loss(observations, actions, discounted_rewards, **kargs)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        opt.apply_gradients(zip(grads, self.model.trainable_weights))