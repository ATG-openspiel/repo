from open_spiel.python import rl_environment
import pyspiel

game = "kuhn_mp_full"
env_configs = {"players": 3}
env = rl_environment.Environment(game, **env_configs)
time_step = env.reset()
num_cards = 3
num_players = 3

def get_imperfect_info_state(time_step):
  current_player = time_step.observations["current_player"]
  origin_info_state = time_step.observations["info_state"][current_player]
  imperfect_info_state = []
  j = 0
  start = num_players + num_cards * current_player
  end = num_players + num_cards * current_player + num_players -1
  for i in range(len(origin_info_state)):
    if i < num_players:
      imperfect_info_state.append(origin_info_state[i])
    # num_players+num_cards*current_player 
    elif i >=  start and i <= end:
      imperfect_info_state.append(origin_info_state[i])
    elif i>= num_players + num_cards * num_players:
      imperfect_info_state.append(origin_info_state[i])
  return imperfect_info_state

def get_legal_cards(time_step):
  """  a list of all the possible cards the teammate may hold. """
  current_player = time_step.observations["current_player"]
  origin_info_state = time_step.observations["info_state"][current_player]
  start = num_players + num_cards * current_player
  end = num_players + num_cards * current_player + num_players 
  cur_card_list = origin_info_state[start:end]
  for i in range(len(cur_card_list)):
    if cur_card_list[i]:
      cur_card = i 
      break 
  legal_cards = []
  for i in range(num_cards):
    if i != cur_card:
      legal_cards.append(i)
  return legal_cards

def get_real_teammate_card(time_step):
  """ return the teammate's real hand card. """
  current_player = time_step.observations["current_player"]
  origin_info_state = time_step.observations["info_state"][current_player]
  if current_player == 1:
    teammate_id = 2
  else:
    teammate_id = 1
  start = num_players + num_cards * teammate_id
  end = num_players + num_cards * teammate_id + num_players 
  teammate_card_list = origin_info_state[start:end]
  for i in range(len(teammate_card_list)):
    if teammate_card_list[i]:
      real_card = i 
      break
  return real_card 


current_player = time_step.observations["current_player"]
print(current_player)
print("perdfect0: ",time_step.observations["info_state"][current_player])
print("imperdfect0: ",get_imperfect_info_state(time_step))

time_step = env.step([1])

current_player = time_step.observations["current_player"]
print(current_player)
print("perdfect1: ",time_step.observations["info_state"][current_player])
print("imperdfect1: ",get_imperfect_info_state(time_step))
print("legal_cards: ",get_legal_cards(time_step))
print("real_card: ",get_real_teammate_card(time_step))

time_step = env.step([1])

current_player = time_step.observations["current_player"]
print(current_player)
print("perdfect2: ",time_step.observations["info_state"][current_player])
print("imperdfect2: ",get_imperfect_info_state(time_step))
print("legal_cards: ",get_legal_cards(time_step))
print("real_card: ",get_real_teammate_card(time_step))



