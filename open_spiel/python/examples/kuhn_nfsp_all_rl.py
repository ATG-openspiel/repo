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

"""NFSP agents trained on Kuhn Poker."""

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import os

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp
import subprocess
  
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(9e8),
                     "Number of training episodes.")
# flags.DEFINE_integer("eval_every", 1000,
#                      "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 10000,
                     "Episode frequency at which the networks are saved.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_string("checkpoint_dir", "model_saved_all_12K4_rl",
                    "Directory to save/load the agent.")

def exec_py(model_load_dir, save_dir, loop_num, num_players, num_cards):
  
  # 定义要执行的Bash命令或脚本
  predict_training_py =  '/repo/open_spiel/python/examples/kuhn_card_predict_example.py'
  equil_calc_py =  '/repo/open_spiel/python/examples/kuhn_equil_use_predict_example_rl.py'
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

  def __init__(self, env, nfsp_policies, mode, num_players):
    game = env.game
    player_ids = [i for i in range(num_players)]
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._num_players = num_players
    self._obs = {"info_state": [None for _ in range(self._num_players)], "legal_actions": [None for _ in range(self._num_players)]}

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


def main(unused_argv): #需要修改原环境中的牌数，与本程序中的人数，和保存位置
  game = "kuhn_mp_full"
  num_players = 3
  num_cards = 4
  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "epsilon_decay_duration": FLAGS.num_train_episodes,
      "epsilon_start": 0.06,
      "epsilon_end": 0.001,
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
                  FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                  **kwargs) for idx in range(num_players)
    ]
    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy, num_players)

    sess.run(tf.global_variables_initializer())
    
    # for agent in agents:
    #   if agent.has_checkpoint(save_dir):
    #     agent.restore(save_dir)
    
    # 将save与eval合并，每次先保存nfsp模型再保存预测模型，再结合两个模型进行评估
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