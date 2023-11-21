import pyspiel

game = pyspiel.load_game("leduc_poker")

#游戏开始
state = game.new_initial_state()

legal_actions = state.legal_actions()
print(legal_actions)

state.apply_action(0)
legal_actions = state.legal_actions()
print(legal_actions)

state.apply_action(2)
legal_actions = state.legal_actions()
print(legal_actions)

state.apply_action(2)
legal_actions = state.legal_actions()
print(legal_actions)

state.apply_action(2)
legal_actions = state.legal_actions()
print(legal_actions)

state.apply_action(1)
legal_actions = state.legal_actions()
print(legal_actions)

state.apply_action(5)
legal_actions = state.legal_actions()
print(legal_actions)
