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

"""Tests for open_spiel.python.algorithms.dqn."""

from absl.testing import absltest
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
import pyspiel
import os
from open_spiel.python.algorithms import exploitability_rl

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

ori_game = "kuhn_poker_mp"
num_players = 3
num_cards = 4
game_name = "12k4"
save_dir = f"/repo/open_spiel/python/examples/model_saved_{game_name}_baseline"
env_configs = {"players": num_players}
env = rl_environment.Environment(ori_game, **env_configs)

# 生成初始随机策略模型并保存
def DQN_self_play():
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
   
  hidden_layers_sizes = [256, 256]  
  replay_buffer_capacity = int(2e5)
  num_train_episodes = int(1e5)
  
  # model_init 随机初始化模型
  with tf.Session() as sess:
    state_representation_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    hidden_layers_sizes = [int(l) for l in hidden_layers_sizes]
    kwargs = {
      "replay_buffer_capacity": replay_buffer_capacity,
      "epsilon_decay_duration": num_train_episodes,
      "epsilon_start": 0.06,
      "epsilon_end": 0.001,
    }
    agents = [
      dqn.DQN(sess, idx, state_representation_size,
                num_actions, hidden_layers_sizes, **kwargs) for idx in range(num_players)
    ]
    sess.run(tf.global_variables_initializer())
      
    baseline = baseline_rl(agents, env, save_dir)
    baseline.DQN_train()
        
        
class baseline_rl(object):

  def __init__(self, base_agents, env, savedir):
    self.base_agents = base_agents
    self.env = env
    # 需要多少个点
    self.total_loop_nums = 100
    # dqn超参数
    self.dqn_loop_nums = 2000
    # self.dqn_save_every = 100
    self.hidden_layers_sizes = [256, 256]
    self.replay_buffer_capacity = int(2e5)
    self.num_train_episodes = int(1e5)
    self.num_players = len(self.base_agents)
    self.game_name = game_name
    self.save_dir = savedir
    self.log_name = f"baseline_{game_name}.log"
    self.log_save_dir = os.path.join(self.save_dir, self.log_name)

  # 依次替换掉导入模型中的每个玩家（用dqn智能体替代），训练得到每个玩家对应源策略的BR策略
  def DQN_train(self):
    for num in range(self.total_loop_nums): 
      for BR_player in range(self.num_players):
        with tf.Session() as sess:
          rl_agent = self.base_agents[BR_player]

          for _ in range(self.dqn_loop_nums):
            time_step = self.env.reset()
            while not time_step.last():
              player_id = time_step.observations["current_player"]
              if player_id == BR_player:
                agent_output = rl_agent.step(time_step)
                time_step = self.env.step([agent_output.action])
              else:
                agent_output = self.base_agents[player_id].step(time_step, is_evaluation=True)
                time_step = self.env.step([agent_output.action])

            rl_agent.step(time_step)

          self.base_agents[BR_player] = rl_agent
          
      # 一次循环结束，计算可利用度    
      exp_rl = exploitability_rl.exploitability_rl(self.base_agents, self.env)    
      expl = exp_rl.exploitability_mp_rl()
      with open(self.log_save_dir, "a") as file:
        file.write(f"{self.game_name} baseline, loop num {(num + 1)*self.dqn_loop_nums}, AVG {expl}---------------------------- \n")
      


  # # 依次替换掉导入模型中的每个玩家（用dqn智能体替代），训练得到每个玩家对应源策略的BR策略
  # def DQN_train(self):
      
  #   for BR_player in range(self.num_players):
  #     with tf.Session() as sess:
  #       rl_agent = self.base_agents[BR_player]

  #       for _ in range(self.dqn_loop_nums):
  #         time_step = self.env.reset()
  #         while not time_step.last():
  #           player_id = time_step.observations["current_player"]
  #           if player_id == BR_player:
  #             agent_output = rl_agent.step(time_step)
  #             time_step = self.env.step([agent_output.action])
  #           else:
  #             agent_output = self.base_agents[player_id].step(time_step, is_evaluation=True)
  #             time_step = self.env.step([agent_output.action])

  #         rl_agent.step(time_step)

  #       self.base_agents[BR_player] = rl_agent
        
  #   # 循环结束，保存模型  
  #   for rl_agent in self.base_agents:
  #     rl_agent.save(self.save_dir)
         

if __name__ == "__main__":
  DQN_self_play()
  
  
