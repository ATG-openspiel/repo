import pyspiel
game = pyspiel.load_game("leduc_poker")
num = 0
for a in range(8):
    for b in range(8):
        if b != a:
            for c in range(8):
                if c != a and c != b:
                    for d in range(8):
                        if d!=a and d!=b and d!=c:
                            print(f"a = {a}, b = {b}, c = {c}, d={d}")
                            state = game.new_initial_state()
                            state.apply_action(a)
                            state.apply_action(b)
                            state.apply_action(c)

                            state.apply_action(2)
                            state.apply_action(2)
                            state.apply_action(1)
                            state.apply_action(1)

                            state.apply_action(d)

                            state.apply_action(1)
                            state.apply_action(2)
                            state.apply_action(1)
                            state.apply_action(2)
                            state.apply_action(1)
                            state.apply_action(2)
                            state.apply_action(1)
                            # print(state.information_state_tensor(0))
                            state.apply_action(1)
                            print(state.information_state_tensor(0))
                            num += 1
print(num)
