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
import copy
from itertools import combinations, permutations

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(9e8),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 100000,
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


# 构造底层游戏环境的上层代理，用来在不修改底层环境的情况下对游戏环境进行改造
class NewKuhn(object):
  
  def __init__(self, game, env_configs,  nfsp_agents, num_players, num_cards):
    self.game = game
    self.env_configs = env_configs
    self.nfsp_agents = nfsp_agents
    self.env = rl_environment.Environment(game, **env_configs)
    self.num_players = num_players
    self.num_cards = num_cards
    
  def observation_spec(self):
    return self.env.observation_spec()
  
  
  # 输入n m, 返回排列Anm的整形列表
  def generate_permutations(self, n, m):
    result = []
    numbers = [int(num) for num in range(1,n+1)]

    # 生成所有三个数字的组合
    combinations_m = combinations(numbers, m)

    # 生成每个组合的排列
    for combination in combinations_m:
        perms = permutations(combination)
        result.extend(perms)

    for i in range(0, len(result)):
      tup = result[i] 
      str_nums = [str(num) for num in tup]
      num_str = ''.join(str_nums)
      num = int(num_str)
      result[i] = num
      
    result = sorted(result)  
    return result

  def reset(self):
    time_step =  self.env.reset()
    player_id = time_step.observations["current_player"]
    ori_info_tensor = time_step.observations['info_state'][player_id]
    hand_card_tensor = ori_info_tensor[self.num_players: self.num_players+self.num_cards]
    legal_actions_list = self.extract_legal_actions(player_id)
    time_step.observations["legal_actions"][player_id] = legal_actions_list
    
    return time_step

  # 从玩家私人手牌向量得到队友可能手牌情况，从而推断出所有合法行动
  # def extract_legal_actions(self, hand_card_tensor, player_id):
  #   actions = self.generate_permutations(self.num_cards, self.num_players-1)
  #   private_card = hand_card_tensor.index(1) + 1
  #   legal_actions = []
  #   for i in range(0, len(actions)):
  #     act = str(actions[i])
  #     if str(private_card) not in act:
  #       legal_actions.append(i)
  #   return legal_actions
  
  # 从玩家私人手牌向量得到队友可能手牌情况，从而推断出所有合法行动
  def extract_legal_actions(self, player_id):
    legal_actions = []
    if player_id == 0:
      for i in range(0, self.num_cards):
        legal_actions.append(i)
    else:
      actions = self.generate_permutations(self.num_cards, self.num_players-1)
      for i in range(0, len(actions)):
        legal_actions.append(i)
          
    return legal_actions
      
  def revise_time_step_by_action(self, time_step):
    player_id = time_step.observations["current_player"]
    ori_info_tensor = time_step.observations['info_state'][player_id]
    actions = self.generate_permutations(self.num_cards, self.num_players-1)
    if player_id == 0:
      current_act = self.action_opponent[0]
      result = [0] * self.num_cards
      result[current_act-1] = 1
      ori_info_tensor[self.num_players:self.num_players + self.num_cards] = result
      start_pos = self.num_players + self.num_cards
      for i in range(self.num_players-1):
        result = [0] * self.num_cards
        ori_info_tensor[start_pos:start_pos] = result
        start_pos += self.num_cards       
    elif player_id > 0:
      current_act = self.action_team[0]
      number_str = str(actions[current_act])
      # 根据动作改造向量
      result = [0] * self.num_cards
      ori_info_tensor[self.num_players:self.num_players + self.num_cards] = result
      start_pos = self.num_players + self.num_cards
      for i in range(len(number_str)):
        result = [0] * self.num_cards
        result[int(number_str[i])-1] = 1
        ori_info_tensor[start_pos:start_pos] = result
        start_pos += self.num_cards 
        
    time_step.observations['info_state'][player_id] = ori_info_tensor
    return time_step 
    
    
  def step(self, action, time_step):
    # time_step = self.env.step(action)
    player_id = time_step.observations["current_player"]
    if player_id == 0:
      self.action_opponent = action
      ori_info_tensor = time_step.observations['info_state'][1]
      hand_card_tensor = ori_info_tensor[self.num_players: self.num_players+self.num_cards]
      legal_actions_list = self.extract_legal_actions(player_id)
      time_step.observations["legal_actions"][1] = legal_actions_list #将返回的time_step中当前玩家合法动作进行修改
      time_step.observations["current_player"] = 1 #将返回的time_step中当前玩家改成团队玩家1
      new_time_step = rl_environment.TimeStep(observations=copy.copy(time_step.observations), 
                                              rewards = [0, 0], 
                                              discounts = copy.copy(time_step.discounts), 
                                              step_type = copy.copy(time_step.step_type))
    elif player_id == 1:
      self.action_team = action 
      time_step.observations["current_player"] = 0 #将返回的time_step中当前玩家改回玩家0，开始整个对局
      time_step.observations["legal_actions"][0] = [0,1]
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        time_step = self.revise_time_step_by_action(time_step)
        # print("time_step: ",time_step,'\n')
        # print("time_step",time_step)
        info_state = time_step.observations['info_state'][player_id]
        legal_actions = time_step.observations["legal_actions"][player_id]
        action, probs = self.nfsp_agents[player_id]._act(info_state, legal_actions)
        # agent_output = self.nfsp_agents[player_id].step(time_step)
        action_list = [action]
        time_step = self.env.step(action_list)
      new_time_step = rl_environment.TimeStep(observations=copy.copy(time_step.observations), 
                                              rewards = [time_step.rewards[0], -1 * time_step.rewards[0]], 
                                              discounts = copy.copy(time_step.discounts), 
                                              step_type = copy.copy(time_step.step_type))
    else:
      print("Illegal_player: ", player_id)
      # print("time_step",time_step)
      
    return new_time_step
    
    
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
  load_game = "kuhn_mp_full"
  saved_game = "kuhn_poker_mp"
  num_players = 3
  num_cards = 4
  env_configs = {"players": num_players}
  load_env = rl_environment.Environment(load_game, **env_configs)
  load_info_state_size = load_env.observation_spec()["info_state"][0]
  load_num_actions = load_env.action_spec()["num_actions"]
  
  # 计算队友手牌可能性（包括与自己手牌重复的不合法情况），队友手牌的所有可能数目即为该玩家动作数量
  saved_num_actions_team = 1
  for i in range (0, num_players - 1):
    saved_num_actions_team *=  (num_cards - i)
    
  saved_num_actions_opponent = num_cards

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "epsilon_decay_duration": FLAGS.num_train_episodes,
      "epsilon_start": 0.06,
      "epsilon_end": 0.001,
  }
  
  current_dir = os.path.dirname(os.path.abspath(__file__))
  # 加载与保存文件名
  load_dir = os.path.join(current_dir, "model_saved_12k4")
  save_dir = os.path.join(current_dir, "model_saved_12k4_alter")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
  with tf.Session() as sess:
    load_nfsp_agents = [
        nfsp.NFSP(sess, idx, load_info_state_size, load_num_actions, hidden_layers_sizes,
                  FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                  **kwargs) for idx in range(num_players)
    ]
    # env = rl_environment.Environment(game, **env_configs)
    saved_env = NewKuhn(saved_game, env_configs, load_nfsp_agents, num_players, num_cards)
    saved_info_state_size = saved_env.observation_spec()["info_state"][0]
    # pylint: disable=g-complex-comprehension
    saved_agents = [
        nfsp.NFSP(sess, 0, saved_info_state_size, saved_num_actions_opponent, hidden_layers_sizes,
                  FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                  **kwargs), 
        nfsp.NFSP(sess, 1, saved_info_state_size, saved_num_actions_team, hidden_layers_sizes,
                  FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                  **kwargs), 
    ]
    
    # todo
    # expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy, num_players)

    # 初始化全局参数
    sess.run(tf.global_variables_initializer())
    
    # 恢复被保存的完美博弈精炼后的nfsp模型参数
    for agent in load_nfsp_agents:
      if agent.has_checkpoint(load_dir):
        agent.restore(load_dir)
    
    # 正式开始训练(保存模型，计算可利用度，对局迭代)
    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.save_every == 0:
        for agent in saved_agents:
          agent.save(save_dir)
      # todo    
      # if (ep + 1) % FLAGS.eval_every == 0:
      #   losses = [agent.loss for agent in agents]
      #   logging.info("Losses: %s", losses)
      #   expl = exploitability.exploitability_mp(env.game, expl_policies_avg)
      #   logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
      #   logging.info("_____________________________________________")

      time_step = saved_env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = saved_agents[player_id].step(time_step)
        # print("time_step_", "__b:  ",saved_agents[player_id]._prev_timestep.observations["info_state"][0][:],'\n')
        action_list = [agent_output.action]
        time_step = saved_env.step(action_list, time_step)

      # Episode is over, step all agents with final info state.
      for agent in saved_agents:
        agent.step(time_step)

if __name__ == "__main__":
  app.run(main)