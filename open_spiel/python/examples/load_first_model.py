import collections
from datetime import datetime
import logging
import os
import random
import sys
import time
from absl import app
from absl import flags
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import pyspiel
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.ppo import PPOAgent
from open_spiel.python.pytorch.ppo import PPOAtariAgent
from open_spiel.python.rl_environment import ChanceEventSampler
from open_spiel.python.rl_environment import Environment
from open_spiel.python.rl_environment import ObservationType
from open_spiel.python.vector_env import SyncVectorEnv


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name",
                    os.path.basename(__file__).rstrip(".py"),
                    "the name of this experiment")
flags.DEFINE_string("game_name", "kuhn_poker", "the id of the OpenSpiel game")
flags.DEFINE_float("learning_rate", 2.5e-4,
                   "the learning rate of the optimizer")
flags.DEFINE_integer("seed", 1, "seed of the experiment")
flags.DEFINE_integer("total_timesteps", 10_000_000,
                     "total timesteps of the experiments")
flags.DEFINE_integer("eval_every", 10, "evaluate the policy every N updates")
flags.DEFINE_bool("torch_deterministic", True,
                  "if toggled, `torch.backends.cudnn.deterministic=False`")
flags.DEFINE_bool("cuda", True, "if toggled, cuda will be enabled by default")


# Algorithm specific arguments
flags.DEFINE_integer("num_envs", 8, "the number of parallel game environments")
flags.DEFINE_integer(
    "num_steps", 128,
    "the number of steps to run in each environment per policy rollout")
flags.DEFINE_bool(
    "anneal_lr", True,
    "Toggle learning rate annealing for policy and value networks")
flags.DEFINE_bool("gae", True, "Use GAE for advantage computation")
flags.DEFINE_float("gamma", 0.99, "the discount factor gamma")
flags.DEFINE_float("gae_lambda", 0.95,
                   "the lambda for the general advantage estimation")
flags.DEFINE_integer("num_minibatches", 4, "the number of mini-batches")
flags.DEFINE_integer("update_epochs", 4, "the K epochs to update the policy")
flags.DEFINE_bool("norm_adv", True, "Toggles advantages normalization")
flags.DEFINE_float("clip_coef", 0.1, "the surrogate clipping coefficient")
flags.DEFINE_bool(
    "clip_vloss", True,
    "Toggles whether or not to use a clipped loss for the value function, as per the paper"
)
flags.DEFINE_float("ent_coef", 0.01, "coefficient of the entropy")
flags.DEFINE_float("vf_coef", 0.5, "coefficient of the value function")
flags.DEFINE_float("max_grad_norm", 0.5,
                   "the maximum norm for the gradient clipping")
flags.DEFINE_float("target_kl", None, "the target KL divergence threshold")



def make_single_env(game_name, seed):

    def gen_env():
        game = pyspiel.load_game(game_name)
        return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))

    return gen_env

# random.seed(FLAGS.seed)
# np.random.seed(FLAGS.seed)
# torch.manual_seed(FLAGS.seed)
# torch.backends.cudnn.deterministic = FLAGS.torch_deterministic
def main(_):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and FLAGS.cuda else "cpu")
    logging.info("Using device: %s", str(device))

    envs = SyncVectorEnv([
        make_single_env(FLAGS.game_name, FLAGS.seed + i)()
        for i in range(FLAGS.num_envs)
    ])
    agent_fn = PPOAgent

    game = envs.envs[0]._game  # pylint: disable=protected-access
    # info_state_shape = game.observation_tensor_shape()
    info_state_shape = [11]



    agent = PPO(
        input_shape=info_state_shape,
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(),
        player_id=0,
        num_envs=FLAGS.num_envs,
        steps_per_batch=FLAGS.num_steps,
        num_minibatches=FLAGS.num_minibatches,
        update_epochs=FLAGS.update_epochs,
        learning_rate=FLAGS.learning_rate,
        gae=FLAGS.gae,
        gamma=FLAGS.gamma,
        gae_lambda=FLAGS.gae_lambda,
        normalize_advantages=FLAGS.norm_adv,
        clip_coef=FLAGS.clip_coef,
        clip_vloss=FLAGS.clip_vloss,
        entropy_coef=FLAGS.ent_coef,
        value_coef=FLAGS.vf_coef,
        max_grad_norm=FLAGS.max_grad_norm,
        target_kl=FLAGS.target_kl,
        device=device,
        agent_fn=agent_fn,
    )

    agent.load_state_dict(torch.load("test_ppo_model.pth"))
    print(agent.state_dict)



if __name__ == "__main__":
    app.run(main)
    