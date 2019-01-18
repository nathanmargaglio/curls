import os
import shutil
import time
import logging
import tensorflow as tf
import numpy as np

from keras.layers import Input, Dense, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K

tf.logging.set_verbosity(tf.logging.ERROR)

class Agent:
    def __init__(self, env, epsilon=0.2, gamma=0.99, entropy_loss=1e-3, actor_lr=0.001, critic_lr=0.005,
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
        self.models = {}

    def build_models(self):
        # Build Actor and Critic models
        self.models['actor'] = self.build_actor_model()
        self.models['critic'] = self.build_critic_model()
        self.DUMMY_ACTION, self.DUMMY_VALUE = np.zeros((1,self.action_space.n)), np.zeros((1,1))

    def proximal_policy_optimization_loss(self, advantage, old_pred, debug=True):

        # Defines the PPO loss to be used during actor optimization
        def loss(y_true, y_pred):
            adv = K.sum(advantage, axis=1)
            prob = y_true * y_pred
            old_prob = y_true * old_pred

            r = K.sum(prob/(old_prob + 1e-10), axis=1)
            clipped = K.clip(r, min_value=1-self.epsilon, max_value=1+self.epsilon)
            minimum = K.minimum(r * adv, clipped * adv)

            entropy_bonus = self.entropy_loss * (prob * K.log(prob + 1e-10))
            entropy_bonus = K.sum(entropy_bonus, axis=1)

            result = -K.mean(minimum + entropy_bonus)
            return result
        return loss

    def build_actor_model(self):
        state_inputs = Input(shape=self.observation_space.shape)
        advantage = Input(shape=(1,))
        old_pred = Input(shape=(self.action_space.n,))

        # hidden layers
        x = Dense(self.hidden_size, activation='relu')(state_inputs)
        x = Dense(self.hidden_size, activation='relu')(x)

        # the output is a probability distribution over the actions
        out_actions = Dense(self.action_space.n, activation='softmax')(x)

        model = Model(inputs=[state_inputs, advantage, old_pred],
                      outputs=[out_actions])

        # compile the model using our custom loss function
        model.compile(optimizer=Adam(lr=self.actor_lr),
                      loss=[self.proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_pred=old_pred
                      )])
        return model

    def build_critic_model(self):
        # critic recieves the observation as input
        state_inputs = Input(shape=self.observation_space.shape)

        # hidden layers
        x = Dense(self.hidden_size, activation='relu')(state_inputs)
        x = Dense(self.hidden_size, activation='relu')(x)

        # we predict the value of the current observation
        predictions = Dense(1, activation='linear')(x)

        model = Model(inputs=state_inputs, outputs=predictions)
        model.compile(optimizer=Adam(lr=self.critic_lr),
                      loss='mse')
        return model

    def step(self, observation):
        # Predict the probability destribution of the actions as a vactor
        prob = self.models['actor'].predict([np.array([observation]),
                                   self.DUMMY_VALUE,
                                   self.DUMMY_ACTION])
        prob = prob.flatten()

        # Sample an action as a scaler
        action = np.random.choice(self.action_space.n, 1, p=prob)[0]

        # Vectorize the action as a one-hot encoding
        action_vector = np.zeros(self.action_space.n)
        action_vector[action] = 1

        return action, action_vector, prob

    def train_on_batch(self, observations, actions, probabilities, rewards):
        # limit our data to the buffer_size
        obs = observations[:self.buffer_size]
        acts = actions[:self.buffer_size]
        probs = probabilities[:self.buffer_size]
        rews = rewards[:self.buffer_size]
        old_probs = probs

        # Calculate advantages
        values = self.models['critic'].predict(obs).reshape((self.buffer_size, 1))
        advs = rews - values

        # Train the actor and critic on the batch data
        self.models['actor'].fit([obs, advs, old_probs], [acts],
                       batch_size=self.batch_size, shuffle=True,
                       epochs=self.epochs, verbose=False)
        self.models['critic'].fit([obs], [advs],
                       batch_size=self.batch_size, shuffle=True,
                        epochs=self.epochs, verbose=False)


    def train(self, episodes=100):
        self.episode = 0
        self.max_episodes = episodes
        self.build_models()
        self.training_loop()

    def training_loop(self):
        # reset the environment
        observations = self.env.reset()
        episode_map = {}

        # Mini batch which contains a single episode's data
        ep_batches = []
        for env_num in range(self.env.num_envs):
            ep_batches.append({
                'observation': [],
                'action_vector': [],
                'probability': [],
                'reward': [],
                'ep_step': 0
            })
            episode_map[env_num] = self.episode
            self.episode += 1


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
                actions = []
                for env_num, observation in enumerate(observations):
                    ep_batch = ep_batches[env_num]

                    # Get the action (scalar), action vector (one-hot vector),
                    # and probability distribution (vector) from the current observation

                    action, action_vector, prob = self.step(observation)
                    actions.append(action)

                    # Append the data to the mini batch
                    ep_batch['observation'].append(observation)
                    ep_batch['action_vector'].append(action_vector)
                    ep_batch['probability'].append(prob)

                next_observations, rewards, dones, infos = self.env.step(actions)
                for env_num, (next_observation, reward, done, info) in enumerate(zip(next_observations, rewards, dones, infos)):
                    ep_batch = ep_batches[env_num]
                    ep_batch['reward'].append(reward)

                # The current observation is now the 'next' observation
                observations = next_observations

                # if the episode is at a terminal state...
                for env_num, done in enumerate(dones):
                    if not done:
                        continue
                    ep_batch = ep_batches[env_num]

                    total_episode_reward = 0
                    for rew in ep_batch['reward']:
                        total_episode_reward += rew

                    print('Ep: {}, Rew: {}'.format(self.episode, total_episode_reward))

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

                    ep_batches[env_num] = {
                        'observation': [],
                        'action_vector': [],
                        'probability': [],
                        'reward': [],
                        'ep_step': -1
                    }

                    # increment the episode count
                    self.episode += 1
                    episode_map[env_num] = self.episode

                # END OF TRAIN STEP
                for ep_batch in ep_batches:
                    ep_batch['ep_step'] += 1
                self.train_step += 1

            # we've filled up our master batch, so we unpack it into numpy arrays
            _observations = np.array(batch['observation'])
            _actions = np.array(batch['action_vector'])
            _probabilities = np.array(batch['probability'])
            _rewards = np.array(batch['reward'])

            # train the agent on the batched data
            self.train_on_batch(_observations, _actions, _probabilities, _rewards)
