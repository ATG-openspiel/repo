import numpy as np
from collections import deque
import torch
import argparse
import random
import glob
from open_spiel.python.pytorch.sac import ReplayBuffer
from open_spiel.python.pytorch.sac import save, collect_random
from open_spiel.python.pytorch.sac import SAC
from open_spiel.python.pytorch.sac import legal_actions_to_mask
import pyspiel
from open_spiel.python.rl_environment import ChanceEventSampler
from open_spiel.python.rl_environment import Environment
from open_spiel.python.rl_environment import ObservationType
from open_spiel.python.vector_env import SyncVectorEnv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time



def make_single_env(game_name, seed):

  def gen_env():
    game = pyspiel.load_game(game_name)
    return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))

  return gen_env


def get_config():
    parser = argparse.ArgumentParser(description='RL') 
    parser.add_argument("--run_name", type=str, default="SAC", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="leduc_poker", help="Gym environment name, default: kuhn_poker(players=3)")
    parser.add_argument("--episodes", type=int, default=3000000, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=200000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=1000, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size, default: 256")
    
    args = parser.parse_args()
    return args


def train(config):

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    current_day = datetime.now().strftime("%d")
    current_month_text = datetime.now().strftime("%h")
    run_name = f"{config.seed}__{current_month_text}__{current_day}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    env = make_single_env(config.env, config.seed)()
    # env.seed(config.seed)
    # env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average100 = deque(maxlen=100)
    total_steps = 0
    
    game = env._game

    player_id = 0
    agent = SAC(state_size=game.information_state_tensor_shape()[0],
                        action_size=game.num_distinct_actions(),
                        device=device, player_id=player_id)

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
    
    collect_random(env=env, dataset=buffer, num_samples=20000, player_id=agent.player_id)
    
    # if config.log_video:
    #     env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

    for i in range(1, config.episodes+1):
        state = env.reset()
        episode_steps = 0
        rewards = 0

        #每局游戏开始时，玩家i行动前其他玩家随机采取动作
        while state.observations["current_player"] != player_id:
            current_player = state.observations["current_player"]
            action = random.choice(state.observations["legal_actions"][current_player])
            state = env.step([action])

        while True:

            #玩家i行动
            state_tensor = torch.Tensor(state.observations["info_state"][agent.player_id])
            la = torch.Tensor(state.observations["legal_actions"][agent.player_id])   
            legal_actions_mask = legal_actions_to_mask(la, game.num_distinct_actions())
            action = agent.get_action(state_tensor, legal_actions_mask)
            steps += 1
            state = env.step([action])

            #玩家i行动后到下次玩家i行动前或玩家i行动后到游戏结束
            while state.observations["current_player"] != player_id:
                if state.last():
                    break
                current_player = state.observations["current_player"]
                action = random.choice(state.observations["legal_actions"][current_player])
                state = env.step([action])#bug

            next_state_tensor = torch.Tensor(state.observations["info_state"][agent.player_id])
            la_next = torch.Tensor(state.observations["legal_actions"][agent.player_id])
            done = state.last()
            buffer.add(state_tensor, la, action, state.rewards[player_id], next_state_tensor, la_next, done)
            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
            
            rewards += state.rewards[player_id]
            episode_steps += 1
            if done:
                break

        average100.append(rewards)
        total_steps += episode_steps
        print("Episode: {} | Reward: {} | Polciy Loss: {} | Alpha: {} | Steps: {}".format(i, rewards, policy_loss, current_alpha, steps))
        
        writer.add_scalar("losses/policy_loss", policy_loss, total_steps)
        writer.add_scalar("losses/bellmann_error1", bellmann_error1, total_steps)
        writer.add_scalar("losses/bellmann_error2", bellmann_error2, total_steps)
        writer.add_scalar("losses/alpha_loss", alpha_loss, total_steps)
        writer.add_scalar("results/alpha", current_alpha, total_steps)
        writer.add_scalar("results/rewards", rewards, total_steps)
        writer.add_scalar("results/average100_rewards", np.mean(average100), total_steps)

        
        # wandb.log({"Reward": rewards,
        #             "Average10": np.mean(average10),
        #             "Steps": total_steps,
        #             "Policy Loss": policy_loss,
        #             "Alpha Loss": alpha_loss,
        #             "Bellmann error 1": bellmann_error1,
        #             "Bellmann error 2": bellmann_error2,
        #             "Alpha": current_alpha,
        #             "Steps": steps,
        #             "Episode": i,
        #             "Buffer size": buffer.__len__()})

        if i % config.save_every == 0:
            save(config, save_name="SAC_discrete", model=agent.actor_local, ep=0)

if __name__ == "__main__":
    config = get_config()
    train(config)

