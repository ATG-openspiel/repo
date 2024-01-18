from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import os

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import handcard_predict
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import exploitability

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


# def get_real_card(time_step):
#   """ return the player's real hand card. """
#   current_player = time_step.observations["current_player"]
#   origin_info_state = time_step.observations["info_state"][current_player]
#   start = 3 + 3 * current_player
#   end = 3 + 3 * current_player + 3
#   teammate_card_list = origin_info_state[start:end]
#   for i in range(len(teammate_card_list)):
#     if teammate_card_list[i]:
#       real_card = i 
#       break
#   return real_card 
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
  
  
def main(unused_argv):
  game = "kuhn_mp_full"
  num_players = 4
  num_cards = 5 #牌数
  
  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
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
  save_dir = os.path.join(current_dir, "model_saved_13k5")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  with tf.Session() as sess:
    nfsp_agents = [
      nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
            FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
            **kwargs) for idx in range(num_players)
    ]
    

    predict_agent = handcard_predict.card_predict(sess, imperfect_info_state_size, num_cards, num_players, hidden_layers_sizes_predict,
                FLAGS.reservoir_buffer_capacity)
    
    sess.run(tf.global_variables_initializer())
    for agent in nfsp_agents:
      if agent.has_checkpoint(save_dir):
        agent.restore(save_dir)
    
    # 通过计算可利用度验证模型是否正确引入
    expl_policies_avg = NFSPPolicies(env, nfsp_agents, nfsp.MODE.average_policy, num_players)
    expl = exploitability.exploitability_mp(env.game, expl_policies_avg)
    logging.info("Saved model exploitability AVG %s", expl)
    
    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.save_every == 0:
          predict_agent.save(save_dir)

      if (ep + 1) % FLAGS.eval_every == 0:
        predict_loss = predict_agent.loss()
        logging.info("Predict loss: %s | predict accuracy: %s at eposide %s.", predict_loss, predict_agent._compute_accuracy(), ep+1)

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if player_id > 0:
          predict_agent.step(time_step)
        info_state = time_step.observations["info_state"][player_id]
        legal_actions = time_step.observations["legal_actions"][player_id]
        action, probs = nfsp_agents[player_id]._act(info_state, legal_actions)
        # if get_real_card(time_step)==0:
        #   action = 1
        # elif get_real_card(time_step)==1:
        #   action = 0
        # elif get_real_card(time_step)==2:
        #   action = 1
          
        time_step = env.step([action])

if __name__ == "__main__":
  app.run(main)

