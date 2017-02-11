import gym
from params import parse_args
from dqn import DQN
import numpy as np
from collections import deque
from datetime import datetime

def run_dqn():
  # get command line arguments, defaults set in utils.py
  agent_params, dqn_params, cnn_params = parse_args()

  env = gym.make(agent_params['environment'])

  episodes = agent_params['episodes']
  steps = agent_params['steps']
  print("# episode to run : {}, # of max steps to run : {}".format(episodes, steps))
  num_actions = env.action_space.n
  observation_shape = env.observation_space.shape

  # initialize dqn learning
  dqn = DQN(num_actions, observation_shape, dqn_params, cnn_params)

  #env = gym.wrappers.Monitor(env, './outputs/cartpole-experiment-' + agent_params['run_id'], force=True)
  last_100 = deque(maxlen=100)

  result_f = open(datetime.now().strftime("data/%Y-%m-%d-%a-%H-%M-%S.csv"), 'w')
  result_f.write("episode, taken_steps, cur_reward, avg_reward")
  for i_episode in range(episodes):
      observation = env.reset()
      reward_sum = 0

      if np.mean(last_100) > steps:
        break

      for t in range(steps):
          #env.render()
          #print observation

          # select action based on the model
          action = dqn.select_action(observation)
          # execute actin in emulator
          new_observation, reward, done, _ = env.step(action)
          # update the state 
          dqn.update_state(action, observation, new_observation, reward, done)
          observation = new_observation

          # train the model
          dqn.train_step()

          reward_sum += reward
          if done:
              break

      print "Episode ", i_episode
      print "Finished after {} timesteps".format(t+1)
      print "Reward for this episode: ", reward_sum
      last_100.append(reward_sum)
      print "Average reward for last 100 episodes: ", np.mean(last_100)
      result_f.write("{},{},{},{}\n".format(i_episode, t+1, reward_sum, np.mean(last_100)))
      result_f.flush()
  result_f.close()

  #env.monitor.close()

if __name__ == '__main__':
  run_dqn()
