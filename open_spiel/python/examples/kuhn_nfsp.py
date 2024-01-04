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

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(9e9),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")

#用于表示NFSP（Neural Fictitious Self-Play）的联合策略
class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = [0, 1, 2]
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {"info_state": [None, None, None], "legal_actions": [None, None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    #print('cur_player:',cur_player)
    legal_actions = state.legal_actions(cur_player)
    #print('legal_actions:',legal_actions)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    #print('self._obs["info_state"][cur_player]:',self._obs["info_state"][cur_player])
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)
    #print('info_state:',info_state)
    
    if cur_player == 0:
        player_id = 0
    else:
        player_id = 1

    #print('self._policies[cur_player]:',self._policies[player_id])
    with self._policies[player_id].temp_mode_as(self._mode):
      p = self._policies[player_id].step(info_state, is_evaluation=True).probs
    
    prob_dict = {action: p[action] for action in legal_actions}
    # print('prob_dict:',prob_dict)
    # print('---------------------------------------------------')
    return prob_dict


def main(unused_argv):
  game = "kuhn_poker_mp"
  num_players = 3
  num_agents = 2

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

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    #创建num_players个NFSP智能体
    agents = [
        nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                  FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                  **kwargs) for idx in range(num_agents)
    ]
    #使用NFSPPolicies类创建了expl_policies_avg，它是一个包含NFSP智能体和环境的策略对象
    #print(agents)
    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    #训练NFSP智能体
    
    #初始化全局变量
    sess.run(tf.global_variables_initializer())
    #在每个训练轮次中
    for ep in range(FLAGS.num_train_episodes):
      #首先检查是否达到了评估的时间点
      if (ep + 1) % FLAGS.eval_every == 0:
        #计算每个智能体的损失（losses），并使用logging.info打印损失信息
        losses = [agent.loss for agent in agents]
        logging.info("Losses: %s", losses)
        #使用exploitability.exploitability函数计算策略expl_policies_avg的可利用性（exploitability），并使用logging.info打印可利用性信息
        #只适用于2人游戏
        #print('-------开始计算可利用度------------')
        expl = exploitability.exploitability_mp(env.game, expl_policies_avg)
        logging.info(expl_policies_avg)
        logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
        logging.info("_____________________________________________")

      #TimeStep对象，其中包含以下内容：
        #observations：刚刚创建的observations字典。
        #rewards：None，表示在第一个时间步没有奖励值。
        #discounts：None，表示在第一个时间步没有折扣值。
        #step_type：StepType.FIRST，表示这是一个新序列的第一个时间步
      #observations字典，用于存储观测信息。字典中包含以下键值对：
        #"info_state"：一个列表，存储每个玩家的信息状态（observation_tensor或information_state_tensor）。
        #"legal_actions"：一个列表，存储每个玩家的合法行动。
        #"current_player"：当前玩家的标识。
        #"serialized_state"：如果_include_full_state为True，则存储游戏和状态的序列化表示。
      time_step = env.reset()

      #使用循环来执行每个时间步的动作选择和智能体的训练
      while not time_step.last():
        #player_id = time_step.observations["current_player"]
        if time_step.observations["current_player"] == 0:
          player_id = 0
        else:
          player_id = 1
        #print('player_id:',player_id)
        agent_output = agents[player_id].step(time_step)
        #print('agent_output:',agent_output)
        action_list = [agent_output.action]
        #print('action_list:',action_list)
        time_step = env.step(action_list)
        #print('time_step:',time_step)
        #print('------------------------------')
      
      #print('——————————————episode 结束——————————————————————')

      # Episode is over, step all agents with final info state.
      #当一个回合（episode）结束后，通过循环迭代所有智能体，并使用最终信息状态（time_step）调用agent.step方法来更新智能体的状态和策略
      for agent in agents:
        agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
