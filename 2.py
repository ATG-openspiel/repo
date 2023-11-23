import pyspiel
import copy
players = 4
game = pyspiel.load_game("leduc_poker(players={})".format(players))
state = game.new_initial_state()

for i in range (0 , players):
    state.apply_action(i)

def isover():
    if players+1 in state.legal_actions():
        return 1
    else:
        return 0
    
max_num = -1
act_num = 0

def dfs(act_num):
    global max_num,state
    if isover():
        if act_num >= max_num:
            max_num = act_num
        return
    for act in state.legal_actions():
        old_state = copy.deepcopy(state)
        state.apply_action(act)
        act_num += 1
        dfs(act_num)
        act_num -= 1
        state = old_state

dfs(act_num)

print(max_num)
