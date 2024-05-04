# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NFSP agents trained on Leduc Poker."""

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability_rl
from open_spiel.python.algorithms import nfsp
import os
import subprocess

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "leduc_mp_full",
                    "Name of the game.")
flags.DEFINE_integer("num_players", 3,
                     "Number of players.")
flags.DEFINE_integer("num_cards", 12,
                     "Number of cards.")
flags.DEFINE_integer("num_train_episodes", int(9e8),
                     "Number of training episodes.")
flags.DEFINE_integer("save_every", 10000,
                     "Episode frequency at which the agents are saved and evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Number of samples in buffer before learning begins.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01,
                   "Learning rate for inner rl agent.")
flags.DEFINE_float("sl_learning_rate", 0.01,
                   "Learning rate for avg-policy sl network.")
flags.DEFINE_string("optimizer_str", "sgd",
                    "Optimizer, choose from 'adam', 'sgd'.")
flags.DEFINE_string("loss_str", "mse",
                    "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_integer("update_target_network_every", 19200,
                     "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for future rewards.")
flags.DEFINE_integer("epsilon_decay_duration", int(20e6),
                     "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.06,
                   "Starting exploration parameter.")
flags.DEFINE_float("epsilon_end", 0.001,
                   "Final exploration parameter.")
flags.DEFINE_string("evaluation_metric", "exploitability",
                    "Choose from 'exploitability', 'nash_conv'.")
flags.DEFINE_bool("use_checkpoints", True, "Save/load neural network weights.")
flags.DEFINE_string("checkpoint_dir", "model_saved_all_12L143_rl",
                    "Directory to save/load the agent.")

def exec_py(model_load_dir, save_dir, loop_num, num_players, num_cards):
  
  # 定义要执行的Bash命令或脚本
  predict_training_py =  '/repo/open_spiel/python/examples/leduc_card_predict_example.py'
  equil_calc_py =  '/repo/open_spiel/python/examples/leduc_equil_use_predict_example_rl.py'
  # 预测信息保存文件名
  predict_log_name = f"predict_{FLAGS.checkpoint_dir[16:]}.log"
  predict_save_dir = os.path.join(save_dir, predict_log_name)
  # 执行 预测网络训练Python 脚本
  process = subprocess.Popen(['python3', predict_training_py, model_load_dir, str(num_players), str(num_cards)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  output, error = process.communicate()
  with open(predict_save_dir, "a") as file:
    file.write(f"{FLAGS.checkpoint_dir[16:]} predict loop num {loop_num}---------------------------- \n")
    # print(f"{FLAGS.checkpoint_dir[16:]} predict loop num {loop_num}----------------------------")
  # print(error.decode('utf-8'),"")
  if output:
    with open(predict_save_dir, "a") as file:
      file.write(output.decode('utf-8'))
      # print(output.decode('utf-8'))
      
  # 最终可利用度信息保存文件名
  exp_log_name = f"exp_{FLAGS.checkpoint_dir[16:]}.log"
  exp_save_dir = os.path.join(save_dir, exp_log_name)
  # 执行 预测网络训练Python 脚本
  process = subprocess.Popen(['python3', equil_calc_py, model_load_dir, str(num_players), str(num_cards)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  output, error = process.communicate()
  with open(exp_save_dir, "a") as file:
    file.write(f"{FLAGS.checkpoint_dir[16:]} exp loop num {loop_num}---------------------------- \n")
    # print(f"{FLAGS.checkpoint_dir[16:]} exp loop num {loop_num}----------------------------")
  print(error.decode('utf-8'),"")
  if output:
    with open(exp_save_dir, "a") as file:
      file.write(output.decode('utf-8'))
      # print(output.decode('utf-8'))
      
class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = list(range(FLAGS.num_players))
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {
        "info_state": [None] * FLAGS.num_players,
        "legal_actions": [None] * FLAGS.num_players
    }

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def main(unused_argv): #需要修改原游戏环境的rank，suit，每回合最大下注数量；以及本程序中的人数，保存路径
  logging.info("Loading %s", FLAGS.game_name)
  game = FLAGS.game_name
  num_players = FLAGS.num_players
  num_cards = FLAGS.num_cards
  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "reservoir_buffer_capacity": FLAGS.reservoir_buffer_capacity,
      "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
      "anticipatory_param": FLAGS.anticipatory_param,
      "batch_size": FLAGS.batch_size,
      "learn_every": FLAGS.learn_every,
      "rl_learning_rate": FLAGS.rl_learning_rate,
      "sl_learning_rate": FLAGS.sl_learning_rate,
      "optimizer_str": FLAGS.optimizer_str,
      "loss_str": FLAGS.loss_str,
      "update_target_network_every": FLAGS.update_target_network_every,
      "discount_factor": FLAGS.discount_factor,
      "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
      "epsilon_start": FLAGS.epsilon_start,
      "epsilon_end": FLAGS.epsilon_end,
  }
  
  current_dir = os.path.dirname(os.path.abspath(__file__))
  # 保存文件名
  save_dir = os.path.join(current_dir, FLAGS.checkpoint_dir)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                  **kwargs) for idx in range(num_players)
    ]
    joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    sess.run(tf.global_variables_initializer())    

    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.save_every == 0:
        dir_with_loop = f"{FLAGS.checkpoint_dir[16:]}_loop{ep+1}"
        save_dir_with_loop = os.path.join(save_dir, dir_with_loop)
        if not os.path.exists(save_dir_with_loop):
          os.makedirs(save_dir_with_loop)
        for agent in agents:
          agent.save(save_dir_with_loop)
          
      # if (ep + 1) % FLAGS.eval_every == 0: 
        exec_py(save_dir_with_loop, save_dir, ep+1, num_players, num_cards)
      

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
