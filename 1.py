
import pyspiel

game = pyspiel.load_game("leduc_poker(players=3)")
state = game.new_initial_state()
state.apply_action(1)
state.apply_action(2)
state.apply_action(3)
state.apply_action(1)
state.apply_action(1)
state.apply_action(2)
state.apply_action(0)
state_copy = game.deserialize_state(state.serialize())
print(state_copy)
