from collections import deque

import gym

from lunar_lander.nn_learning_agent import LearningAgent

# Gym settings
max_episodes = 50000

# Episode rewards queue
rewards = deque(maxlen=100)

# Recording settings
record = True
recording_file = '/tmp/lunar_lander'
key = 'sk_WqHacrBQcaSmyTwOqpLtg'

# Init Lunar Lander Gym
env = gym.make('LunarLander-v2')
if record:
    env.monitor.start(recording_file, force=True)
env.reset()

# Agent
agent = LearningAgent()

# Agent is consider trained and should stop additional learning once the average
# reward over the last 100 episodes exceeds the stop_at value.
stop_at = 200

# Agent will train less frequently once performing well to help avoid over
# fitting, aka freeze.
freeze_at = 100
frozen_training_rate = 0.1

# Begin running gym episodes
for i_episode in range(max_episodes):
    observation = env.reset()
    done = False

    # Reward total for the episode and current time step
    episode_reward = 0
    time_step = 0

    # Let agent complete the episode, force end the episode if it last longer
    # than 100000 time steps (if it gets stuck)
    while not done and time_step < 100000:
        # Backup old observation
        old_observation = observation

        # Select action to take
        action = agent.act(old_observation)

        # Take action
        observation, reward, done, info = env.step(action)
        time_step += 1
        episode_reward += reward
        agent.store_in_replay(old_observation, action, reward, observation,
                              done)

    # Append episode reward, take running average
    rewards.append(episode_reward)
    avg_reward = sum(rewards) / len(rewards)

    if avg_reward >= freeze_at:
        agent.set_attributes({'training_rate': frozen_training_rate})

    # Stop all training, and set epsilon to 0 if threshold has been reached
    if avg_reward >= stop_at:
        agent.stop_training()

    # Log some stats to standard out
    if i_episode > 100:
        STATS_FMT = "Ep: {}, Avg: {}, Cur: {}"
        print(STATS_FMT.format(i_episode, avg_reward, episode_reward))

# Upload recording to OpenAI
if record:
    env.monitor.close()
    gym.upload(recording_file, api_key=key)
