from collections import deque

import gym

from lunar_lander.nn_learning_agent import LearningAgent

# Gym settings
max_episodes = 25000

# Episode rewards queue
rewards = deque(maxlen=100)

# Recording settings
record = True
recording_file = '/tmp/lander_test_123'
key = 'sk_WqHacrBQcaSmyTwOqpLtg'

# Init Lunar Lander Gym
env = gym.make('LunarLander-v2')
if record:
    env.monitor.start(recording_file, force=True)
env.reset()

# Agent
agent = LearningAgent()

# Exploring settings
action_time_step_range = 10
exploring = True
exploring_episodes = 25

# Freeze training after every action, and instead train after every episode
# to prevent over fitting once the average reward over the last 100 episodes
# exceeds the freeze_at value.
freeze = False
freeze_at = 100

# Agent is consider trained and should stop additional learning once the average
# reward over the last 100 episodes exceeds the stop_at value.
stop = False
stop_at = 185

# Begin running gym episodes
for i_episode in range(max_episodes):
    observation = env.reset()
    done = False

    # Reward total for the episode and current time step
    episode_reward = 0
    time_step = 0

    # Turn off pure exploration mode once threshold is reached
    if i_episode > exploring_episodes:
        exploring = False

    # If not exploring, a new action should be chosen at every time step
    if not exploring:
        action_time_step_range = 1

    # Let agent complete the episode, force end the episode if it last longer
    # than 1000 time steps
    while not done and time_step < 1000:

        old_observation = observation

        action = agent.get_action(old_observation)

        # Repeat selected action for action_time_step_range time steps
        for i in range(action_time_step_range):
            observation, reward, done, info = env.step(action)
            time_step += 1
            episode_reward += reward
            agent.store_in_replay(old_observation, action, reward, observation,
                                  done)
            if done:
                break

        # If not in freeze or exploring mode, agent should train each time step
        if not exploring and not freeze:
            agent.train()

    rewards.append(episode_reward)
    avg_reward = sum(rewards) / len(rewards)

    # Enable freeze mode is threshold has been reached
    if avg_reward >= freeze_at:
        freeze = True

    # Stop all training, and set epsilon to 0 if threshold has been reached
    if avg_reward >= stop_at:
        agent.no_eps()
        stop = True

    # If per time step training is frozen, train after the episode
    if freeze and not stop:
        agent.train()

    # Log some stats to standard out
    if i_episode > 100:
        STATS_FMT = "Ep: {}, Avg: {}, Cur: {}, Freeze: {}, Stop: {}"
        print(STATS_FMT.format(i_episode, avg_reward, episode_reward, freeze,
                               stop))

if record:
    env.monitor.close()
    gym.upload(recording_file, api_key=key)
