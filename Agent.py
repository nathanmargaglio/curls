import os
import shutil
import time
import logging
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

class ActorModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(100, activation='relu')
        self.policy_logits = layers.Dense(action_size)

    def call(self, observation):
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        # Forward pass
        x = self.dense1(observation)
        logits = self.policy_logits(x)
        return logits
    
class CriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(CriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense2 = layers.Dense(100, activation='relu')
        self.values = layers.Dense(1)

    def call(self, observation):
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        # Forward pass
        v1 = self.dense2(observation)
        values = self.values(v1)
        return values

class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.actor = ActorModel(state_size, action_size)
        self.critic = CriticModel(state_size, action_size)

    def call(self, observation):
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        # Forward pass
        logits = self.actor(observation)
        values = self.critic(observation)
        return logits, values

class Agent:
    def __init__(self, env, epsilon=0.2, gamma=0.99, entropy_loss=1e-3, actor_lr=0.001, critic_lr=0.001,
                hidden_size=128, epochs=10, batch_size=64, buffer_size=256, *args, **kwargs):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # Set hyperparameters
        self.epsilon = epsilon
        self.gamma = gamma
        self.entropy_loss = entropy_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.model = ActorCriticModel(self.observation_space.shape, self.action_space.n)

    def proximal_policy_optimization_loss(self, advantage, old_pred):

        # Defines the PPO loss to be used during actor optimization
        def loss(y_true, y_pred):
            adv = tf.reduce_sum(advantage, axis=1)
            adv = tf.cast(adv, tf.float32)
            prob = y_true * y_pred
            old_prob = y_true * old_pred

            r = tf.reduce_sum(prob/(old_prob + 1e-10), axis=1)
            r = tf.cast(r, tf.float32)
            
            clipped = tf.clip_by_value(
                r,
                clip_value_min = 1 - self.epsilon,
                clip_value_max = 1 + self.epsilon
            )
            minimum = tf.minimum(r * adv, clipped * adv)

            entropy_bonus = self.entropy_loss * (prob * tf.log(prob + 1e-10))
            entropy_bonus = tf.reduce_sum(entropy_bonus, axis=1)
            entropy_bonus = tf.cast(entropy_bonus, tf.float32)
            
            result = - tf.reduce_mean(minimum + entropy_bonus)
            return result
        return loss

    def step(self, observation):
        # Predict the probability destribution of the actions as a vactor
        prob, _ = self.model(np.array([observation]))
        prob = tf.nn.softmax(prob).numpy().flatten()

        # Sample an action as a scaler
        action = np.random.choice(self.action_space.n, 1, p=prob)[0]

        # Vectorize the action as a one-hot encoding
        action_vector = np.zeros(self.action_space.n)
        action_vector[action] = 1

        return action, action_vector, prob
    
    def get_grads(self, observation, action, advantage, old_pred):
        observation_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(action, dtype=tf.float32)
        advantage_tensor = tf.convert_to_tensor(advantage, dtype=tf.float32)
        old_pred_tensor = tf.convert_to_tensor(old_pred, dtype=tf.float32)
        
        # Actor Grads
        loss_function = self.proximal_policy_optimization_loss(advantage_tensor, old_pred_tensor)
        with tf.GradientTape() as tape:
            logits, values = self.model(observation_tensor)
            probs = tf.nn.softmax(logits)
            actor_loss = -loss_function(action_tensor, probs)
            #actor_loss = values*tf.log(probs + 1e-10)
            
        actor_grad_tensors = tape.gradient(actor_loss, self.model.actor.trainable_weights)
        actor_grads = [g.numpy() for g in actor_grad_tensors]
        
        with tf.GradientTape() as tape:
            logits, values = self.model(observation_tensor)
            probs = tf.nn.softmax(logits)
            critic_loss = -tf.reduce_sum((advantage_tensor - values)**2)
            
        critic_grad_tensors = tape.gradient(critic_loss, self.model.critic.trainable_weights)
        critic_grads = [g.numpy() for g in critic_grad_tensors]

        return actor_grads, critic_grads
    
    def apply_grads(self, actor_grads, critic_grads):
        for w, ag in zip(self.model.actor.trainable_weights, actor_grads):
            w.assign_add(self.actor_lr * np.array(ag))
            
        for w, cg in zip(self.model.critic.trainable_weights, critic_grads):
            w.assign_add(self.critic_lr * np.array(cg))

    def train_on_batch(self, observations, actions, probabilities, rewards):
        # limit our data to the buffer_size
        obs = observations[:self.buffer_size]
        acts = actions[:self.buffer_size]
        probs = probabilities[:self.buffer_size]
        rews = rewards[:self.buffer_size]
        old_probs = probs
        
        # Calculate advantages
        _, values = self.model(obs)
        advs = rews - values.numpy().reshape((self.buffer_size, 1))
        
        for epoch in range(self.epochs):
            batch_index = np.random.choice(self.buffer_size, self.batch_size)
            actor_grads, critic_grads = self.get_grads(
                obs[batch_index],
                acts[batch_index],
                advs[batch_index],
                old_probs[batch_index]
            )
            self.apply_grads(actor_grads, critic_grads)

    def train(self, episodes=100):
        self.episode = 0
        self.max_episodes = episodes
        self.training_loop()

    def training_loop(self):
        self.episode_rews = []
        # reset the environment
        observation = self.env.reset()

        # Mini batch which contains a single episode's data
        ep_batch = {
            'observation': [],
            'action_vector': [],
            'probability': [],
            'reward': [],
            'ep_step': 0
        }

        self.train_step = 0
        # Collect a batch of samples
        while self.episode < self.max_episodes:
            # 'Master Batch' that we add mini batches to
            batch = {
                'observation': [],
                'action_vector': [],
                'probability': [],
                'reward': []
            }

            # While we don't hit the buffer size with our master batch...
            while len(batch['observation']) < self.buffer_size:
                # Get the action (scalar), action vector (one-hot vector),
                # and probability distribution (vector) from the current observation

                action, action_vector, prob = self.step(observation)

                next_observation, reward, done, info = self.env.step(action)
                
                # Append the data to the mini batch
                ep_batch['observation'].append(observation)
                ep_batch['action_vector'].append(action_vector)
                ep_batch['probability'].append(prob)
                ep_batch['reward'].append(reward)

                # The current observation is now the 'next' observation
                observation = next_observation

                # if the episode is at a terminal state...
                if done:
                    total_episode_reward = 0
                    for rew in ep_batch['reward']:
                        total_episode_reward += rew

                    if (self.episode + 1) % 100 == 0:
                        print('Ep: {}, Rew: {}'.format(self.episode, total_episode_reward))
                    self.episode_rews.append(total_episode_reward)

                    # transform rewards based to discounted cumulative rewards
                    for j in range(len(ep_batch['reward']) - 2, -1, -1):
                        ep_batch['reward'][j] += ep_batch['reward'][j + 1] * self.gamma

                    # for each entry in the mini batch...
                    for i in range(len(ep_batch['observation'])):
                        # we unpack the data
                        obs = ep_batch['observation'][i]
                        act = ep_batch['action_vector'][i]
                        prob = ep_batch['probability'][i]
                        r = ep_batch['reward'][i]

                        # and pack it into the master batch
                        batch['observation'].append(obs)
                        batch['action_vector'].append(act)
                        batch['probability'].append(prob)
                        batch['reward'].append([r])

                    # reset the environment
                    # observations = self.env.reset()

                    ep_batch = {
                        'observation': [],
                        'action_vector': [],
                        'probability': [],
                        'reward': [],
                        'ep_step': -1
                    }

                    observation = self.env.reset()
                    # increment the episode count
                    self.episode += 1

                # END OF TRAIN STEP
                ep_batch['ep_step'] += 1
                self.train_step += 1

            # we've filled up our master batch, so we unpack it into numpy arrays
            _observations = np.array(batch['observation'])
            _actions = np.array(batch['action_vector'])
            _probabilities = np.array(batch['probability'])
            _rewards = np.array(batch['reward'])

            # train the agent on the batched data
            self.train_on_batch(_observations, _actions, _probabilities, _rewards)