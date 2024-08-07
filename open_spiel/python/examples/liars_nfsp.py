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

"""NFSP agents trained on Goofspiel."""

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import os

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability_rl
from open_spiel.python.algorithms import nfsp

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(9e8),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 20000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 20000,
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
  game = "liars_dice_mp"
  num_players = 3 #玩家人数
  num_dices = 1     #每人骰子数
  num_dice_sides = 3  #骰子面数

  env_configs = {"players": num_players,
                 "numdice":num_dices,
                 "dice_sides":num_dice_sides}
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
  game_name = "12D13"
  save_dir = os.path.join(current_dir, f"model_saved_{game_name}_baseline_3")
  
  exp_log_name = f"baseline_{game_name}.log"
  log_save_dir = os.path.join(save_dir, exp_log_name)
  
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
    
    for agent in agents:
      if agent.has_checkpoint(save_dir):
        agent.restore(save_dir)
        
    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.save_every == 0:
        for agent in agents:
          agent.save(save_dir)
          
      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        logging.info("Losses: %s", losses)
        exp_rl = exploitability_rl.exploitability_rl(agents, env)
        # expl = exp_rl.exploitability_mp_rl()
        with open(log_save_dir, "a") as file:
          file.write(f"{game_name} baseline loop num {ep + 1}---------------------------- \n")
        # expl = exploitability.exploitability_mp_out(env.game, expl_policies_avg, log_save_dir)
        expl = exp_rl.exploitability_mp_rl_out(log_save_dir)
        with open(log_save_dir, "a") as file:
          file.write(f"TMECor baseline Exploitability AVG {expl}\n")

      time_step = env.reset()
      #print(time_step)
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        #print(player_id)
        agent_output = agents[player_id].step(time_step)
        #print(agent_output)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)


if __name__ == "__main__":
  app.run(main)