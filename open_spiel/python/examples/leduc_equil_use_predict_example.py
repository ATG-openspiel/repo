from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import os
import itertools
import sys
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import leduc_handcard_predict
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import exploitability

args = sys.argv[1:]
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(3e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 1000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 10000,
                     "Episode frequency at which the networks are saved.")
flags.DEFINE_list("hidden_layers_sizes_predict", [
    256, 256, 
], "Number of hidden units in the predict net.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")


class Predict_NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, predict_net, mode, num_cards, num_players):
    game = env.game
    player_ids = [i for i in range(num_players)]
    super(Predict_NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._predict_net = predict_net
    self._mode = mode
    self._num_cards = num_cards
    self._num_players = num_players
    self._obs = {"info_state": [None for _ in range(self._num_players)], "legal_actions": [None for _ in range(self._num_players)]}
    self._output_cards_list = self.generate_all_possible_cards(self._num_cards, self._num_players - 2)
    
  # 生成队友手牌的所有可能排列情况
  def generate_all_possible_cards(self, cards, num_teammates):
    cards_list = [i for i in range(cards)]
    permutations = list(itertools.permutations(cards_list, r=num_teammates))
    permutations_as_lists = [list(perm) for perm in permutations]
    return permutations_as_lists
  
  
  def rebuild_infostate_with_card_predict(self, origin_info_state, card):
    """ return  info_state_with_card_predict. """
    
    current_player = self._obs["current_player"]
    infostate_with_card_predict = []
    for i in range(len(origin_info_state)):
      if i < self._num_players:
        infostate_with_card_predict.append(origin_info_state[i])

      elif i == self._num_players:
        if current_player == 0:
          for m in range(self._num_players):
            for n in range(self._num_cards):
              if m == 0:
                infostate_with_card_predict.append(origin_info_state[i + n])
              else:
                infostate_with_card_predict.append(0.)
        else:
          flag = 0
          for m in range(self._num_players):
            for n in range(self._num_cards):
              if m == 0:
                infostate_with_card_predict.append(0.)
              elif m == current_player:
                infostate_with_card_predict.append(origin_info_state[i + n])
              else:
                if n == self._output_cards_list[card][flag]:
                  infostate_with_card_predict.append(1.)
                else:
                  infostate_with_card_predict.append(0.)
            if m != 0 and m != current_player:
              flag += 1
                          
      elif i>= self._num_players + self._num_cards:
        infostate_with_card_predict.append(origin_info_state[i])
    return infostate_with_card_predict
      
      
  def get_legal_cards(self, time_step):
    """  a list of all the possible cards the teammate may hold. 
    
      Args:
        time_step: a time step with imperfect infostate tensor.
    """
    current_player = time_step.observations["current_player"]
    origin_info_state = time_step.observations["info_state"][current_player]
    private_start = self._num_players
    private_end = self._num_players + self._num_cards
    public_start = self._num_players + self._num_cards
    public_end = self._num_players + self._num_cards + self._num_cards
    cur_card_list = origin_info_state[private_start:private_end]
    pub_card_list = origin_info_state[public_start:public_end]
    
    for i in range(len(cur_card_list)):
      if cur_card_list[i]:
        cur_card = i 
        break 
      
    pub_card = -1
    for i in range(len(pub_card_list)):
      if pub_card_list[i]:
        pub_card = i 
        break 
      
    if pub_card >= 0:
      legal_cards = [i for i, sublist in enumerate(self._output_cards_list) if cur_card not in sublist and pub_card not in sublist]
    else:
      legal_cards = [i for i, sublist in enumerate(self._output_cards_list) if cur_card not in sublist]

    return legal_cards
  
  
  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    ori_infostate = state.information_state_tensor(cur_player)

    self._obs["info_state"][cur_player] = ori_infostate
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)
    
    predict_cards, _, _ = self._predict_net._predict(ori_infostate, self.get_legal_cards(info_state))
    infostate_with_card_predict = self.rebuild_infostate_with_card_predict(state.information_state_tensor(cur_player), predict_cards)
    info_state.observations["info_state"][cur_player] = (infostate_with_card_predict)
    
    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict
  

def main(unused_argv): #需要修改人数，牌数，保存路径
  ori_game = "leduc_poker_mp"
  full_game = "leduc_mp_full"
  num_players = int(args[1])
  num_cards = int(args[2]) #牌数
  
  env_configs = {"players": num_players}
  env = rl_environment.Environment(full_game, **env_configs)
  ori_env = rl_environment.Environment(ori_game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  imperfect_info_state_size = info_state_size - (num_players - 1) * num_cards #非完全信息info_state_size
  num_actions = env.action_spec()["num_actions"]
  
  hidden_layers_sizes_predict = [int(l) for l in FLAGS.hidden_layers_sizes_predict]
  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "epsilon_decay_duration": FLAGS.num_train_episodes,
      "epsilon_start": 0.06,
      "epsilon_end": 0.001,
  }
  
  current_dir = os.path.dirname(os.path.abspath(__file__))
  # 保存路径名
  # save_dir = os.path.join(current_dir, "model_saved_12L143")
  
  save_dir = args[0]
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  with tf.Session() as sess:
    nfsp_agents = [
      nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
            FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
            **kwargs) for idx in range(num_players)
    ]
    
    for agent in nfsp_agents:
      # if agent.has_checkpoint(save_dir):
      agent.restore(save_dir)
    
    predict_agent = leduc_handcard_predict.card_predict(sess, imperfect_info_state_size, num_cards, num_players, hidden_layers_sizes_predict,
                FLAGS.reservoir_buffer_capacity)
    
    # if predict_agent.has_checkpoint(save_dir):
    predict_agent.restore(save_dir)
      

    expl_policies_avg = Predict_NFSPPolicies(ori_env, nfsp_agents, predict_agent, nfsp.MODE.average_policy, num_cards, num_players)
    
    expl = exploitability.exploitability_mp(ori_env.game, expl_policies_avg)
    print(f"TMECor with predict exploitability AVG {expl}")
      


    

if __name__ == "__main__":
  app.run(main)