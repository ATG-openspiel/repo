from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import os
import itertools
import sys
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import liars_handcard_predict
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import exploitability_rl

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


class Predict_NFSPPolicies_Agents(object):
  """Joint policy to be evaluated."""

  def __init__(self, nfsp_agents, predict_net, num_dices, num_dice_sides, num_players):
    self.player_ids = [i for i in range(num_players)]
    self._ori_agents = nfsp_agents
    self._predict_net = predict_net
    self._num_players = num_players
    self._num_dices = num_dices
    self._num_dice_sides = num_dice_sides
    self._num_cards = num_dices * num_dice_sides
    self._output_cards_list = self.generate_all_possible_cards(self._num_cards, self._num_players - 2)
    
    
  def __len__(self):
        return len(self._ori_agents)
    
    
  def __getitem__(self, index):
        return self
      
      
  # 生成队友手牌的所有可能排列情况, 举例来看若游戏为423，则每个人手牌有6种可能，生成的排列为[[1,1],[1,2].....[6,6]]
  def generate_all_possible_cards(self, cards, num_teammates):
    cards_list = [i for i in range(cards)]
    permutations = list(itertools.product(cards_list, repeat=num_teammates))
    permutations_as_lists = [list(perm) for perm in permutations]
    return permutations_as_lists
  
  
  # 从信息集向量中提取当前玩家
  def get_current_player(self, origin_info_state):
    players_tensor = origin_info_state[0:self._num_players]
    current_player = next(j for j, bit in enumerate(players_tensor) if bit)
    return current_player
    
  
  def bit_add(self, num_str, pos):
    cur_bit = int(num_str[pos])
    if cur_bit == self._num_dice_sides:
      num_str[pos] = '1'
      bit_add(num_str, pos-1)
      
    else:
      cur_bit = int(num_str[pos]) + 1
      num_str[pos] = str(cur_bit)
      
    return num_str
    
    
  def index_transfer(self, num):
    result = 1
    for _ in range(self._num_dices-1):
      result = result*10 + 1
    
    result = list(str(result))
    for _ in range(num - 1):
      result = self.bit_add(result, -1)
      
    return result
      
    
    
  def rebuild_infostate_with_card_predict(self, origin_info_state, card):
    """ return  info_state_with_card_predict. """
    
    current_player = self.get_current_player(origin_info_state)
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
                tmp = self._output_cards_list[card][flag]
                # print("tmp1 ",tmp)
                tmp = self.index_transfer(tmp)
                # print("tmp2 ",tmp)
                for j in range(len(tmp)):
                  bit = tmp[-1-j]
                  for i in range(1,self._num_dice_sides+1):
                    if i == bit:
                      infostate_with_card_predict.append(1.)
                    else:
                      infostate_with_card_predict.append(0.)
                break       

            if m != 0 and m != current_player:
              flag += 1
                          
      elif i>= self._num_players + self._num_cards:
        infostate_with_card_predict.append(origin_info_state[i])
    return infostate_with_card_predict
      
      
  def get_legal_cards(self):
    """  a list of all the possible cards indexes of the output_cards_list which the teammates may hold. 
    
      Args:
        time_step: a time step with full infostate tensor.
    """
    # current_player = time_step.observations["current_player"]
    # origin_info_state = time_step.observations["info_state"][current_player]
    # start = self._num_players + self._num_cards * current_player
    # end = self._num_players + self._num_cards * current_player + self._num_cards
    # cur_card_list = origin_info_state[start:end]
    # for i in range(len(cur_card_list)):
    #   if cur_card_list[i]:
    #     cur_card = i 
    #     break 
    legal_cards = [i for i, sublist in enumerate(self._output_cards_list)]
    return legal_cards
  
  
  def _act(self, info_state, legal_actions):
    cur_player = self.get_current_player(info_state)
    
    predict_cards, _, _ = self._predict_net._predict(info_state, self.get_legal_cards())
    # predict_cards是预测队友卡牌数组的索引
    infostate_with_card_predict = self.rebuild_infostate_with_card_predict(info_state, predict_cards)
    # print("infostate_with_card_predict ",infostate_with_card_predict)

    action, probs = self._ori_agents[cur_player]._act(infostate_with_card_predict, legal_actions)
    return action, probs
  

def main(unused_argv): #需要修改人数，牌数，保存路径
  ori_game = "liars_dice_mp"
  full_game = "liars_dice_info"
  num_players = 3 #玩家人数
  numdice = 1     #每人骰子数
  dice_sides = 3  #骰子面数
  num_cards = numdice * dice_sides
  
  env_configs = {"players": num_players}
  env = rl_environment.Environment(full_game, **env_configs)
  ori_env = rl_environment.Environment(ori_game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  imperfect_info_state_size = info_state_size - (num_players - 1) * num_cards #非完全信息info_state_size
  # print("info_state_size: ",info_state_size)
  # print("imperfect_info_state_size: ",imperfect_info_state_size)
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
  save_dir = os.path.join(current_dir, "model_saved_12D13")
  # save_dir = args[0]
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  with tf.Session() as sess:
    nfsp_agents = [
      nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
            FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
            **kwargs) for idx in range(num_players)
    ]
    
    for agent in nfsp_agents:
      if agent.has_checkpoint(save_dir):
        agent.restore(save_dir)
    
    predict_agent = liars_handcard_predict.card_predict(sess, imperfect_info_state_size, numdice, dice_sides,  num_players, hidden_layers_sizes_predict,
                FLAGS.reservoir_buffer_capacity)
    
    if predict_agent.has_checkpoint(save_dir):
      predict_agent.restore(save_dir)
      
    our_agents = Predict_NFSPPolicies_Agents(nfsp_agents, predict_agent, numdice, dice_sides, num_players) 
    exp_rl = exploitability_rl.exploitability_rl(our_agents, ori_env)
    expl = exp_rl.exploitability_mp_rl()
    
    # expl_policies_avg = Predict_NFSPPolicies(ori_env, nfsp_agents, predict_agent, nfsp.MODE.average_policy, num_cards, num_players) 
    # expl = exploitability.exploitability_mp(ori_env.game, expl_policies_avg)
    
    print(f"TMECor with predict exploitability AVG {expl}")
      


    

if __name__ == "__main__":
  app.run(main)