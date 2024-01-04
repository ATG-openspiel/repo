# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural Fictitious Self-Play (NFSP) agent implemented in TensorFlow.

See the paper https://arxiv.org/abs/1603.01121 for more details.
"""

import collections
import contextlib
import enum
import os
import random
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.algorithms import dqn

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition", "info_state action_probs legal_actions_mask")

MODE = enum.Enum("mode", "best_response average_policy")


class NFSP(rl_agent.AbstractAgent):
  """NFSP Agent implementation in TensorFlow.

  See open_spiel/python/examples/kuhn_nfsp.py for an usage example.
  """

  def __init__(self,
               session,
               player_id,
               state_representation_size,
               num_actions,
               hidden_layers_sizes,
               reservoir_buffer_capacity,#用于经验回放的储存缓冲区的容量
               anticipatory_param,
               batch_size=128,#用于训练神经网络的小批量大小
               rl_learning_rate=0.001,#强化学习更新的学习率
               sl_learning_rate=0.01,#监督学习更新的学习率
               min_buffer_size_to_learn=1000,
               learn_every=64,
               optimizer_str="sgd",#用于训练神经网络的优化器（默认为随机梯度下降）
               **kwargs):
    """Initialize the `NFSP` agent."""
    self.player_id = player_id #该代理控制的玩家的标识符
    self._session = session #用于训练代理的TensorFlow会话
    self._num_actions = num_actions #环境中可能的动作数量
    self._layer_sizes = hidden_layers_sizes #一个包含隐藏层大小的整数列表，表示神经网络的结构
    self._batch_size = batch_size#存储批量大小
    self._learn_every = learn_every#存储学习间隔
    self._anticipatory_param = anticipatory_param#存储探索与利用之间的参数
    self._min_buffer_size_to_learn = min_buffer_size_to_learn#存储进行学习所需的最小缓冲区大小

    self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)#初始化一个用于经验回放的储存缓冲区对象
    self._prev_timestep = None#存储上一个时间步
    self._prev_action = None#存储上一个动作

    # Step counter to keep track of learning.
    #步骤计数器，用于跟踪学习的进展。它被初始化为0，并在代理的训练过程中逐步递增。
    self._step_counter = 0

    # Inner RL agent
    kwargs.update({
        "batch_size": batch_size,
        "learning_rate": rl_learning_rate,
        "learn_every": learn_every,
        "min_buffer_size_to_learn": min_buffer_size_to_learn,
        "optimizer_str": optimizer_str,
    })
    #使用深度Q网络（DQN）来实现强化学习，并在训练过程中学习和优化策略。
    self._rl_agent = dqn.DQN(session, player_id, state_representation_size,
                             num_actions, hidden_layers_sizes, **kwargs)

    # Keep track of the last training loss achieved in an update step.
    #跟踪上一次更新步骤中达到的强化学习和监督学习损失值
    self._last_rl_loss_value = lambda: self._rl_agent.loss
    self._last_sl_loss_value = None

    # Placeholders.
    #占位符，用于接收信息状态的输入
    self._info_state_ph = tf.placeholder(
        shape=[None, state_representation_size],#None表示可以接受任意数量的信息状态，state_representation_size表示每个信息状态的尺寸
        dtype=tf.float32,
        name="info_state_ph")

    #占位符，用于接收动作概率的输入。
    self._action_probs_ph = tf.placeholder(
        shape=[None, num_actions], #None表示可以接受任意数量的动作概率，num_actions表示动作的数量
        dtype=tf.float32, 
        name="action_probs_ph")

    #占位符，用于接收合法动作的掩码输入
    self._legal_actions_mask_ph = tf.placeholder(
        shape=[None, num_actions],#None表示可以接受任意数量的掩码，num_actions表示动作的数量
        dtype=tf.float32,
        name="legal_actions_mask_ph")

    # Average policy network.
    #创建了一个平均策略网络

    #创建了一个多层感知机（MLP）模型，用于表示平均策略网络。
    #这个多层感知机的输入大小为state_representation_size，隐藏层的结构由self._layer_sizes定义
    #输出层的大小为num_actions，即动作的数量
    #这个平均策略网络用于生成平均策略的动作概率。
    self._avg_network = simple_nets.MLP(state_representation_size,
                                        self._layer_sizes, num_actions)
    #通过将信息状态占位符self._info_state_ph传递给平均策略网络，计算得到平均策略的输出self._avg_policy
    self._avg_policy = self._avg_network(self._info_state_ph)
    #通过应用softmax函数，将这些输出转换为动作概率分布，保存在self._avg_policy_probs中
    self._avg_policy_probs = tf.nn.softmax(self._avg_policy)

    self._savers = [
        ("q_network", tf.train.Saver(self._rl_agent._q_network.variables)),#用于保存强化学习代理的Q网络的变量（即self._rl_agent._q_network.variables）
        ("avg_network", tf.train.Saver(self._avg_network.variables))#保存平均策略网络的变量（即self._avg_network.variables）
    ]

    # Loss
    #损失函数被定义为平均交叉熵损失
    self._loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(self._action_probs_ph),
            logits=self._avg_policy))

    if optimizer_str == "adam":
      optimizer = tf.train.AdamOptimizer(learning_rate=sl_learning_rate)
    elif optimizer_str == "sgd":#梯度下降优化器
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=sl_learning_rate)
    else:
      raise ValueError("Not implemented. Choose from ['adam', 'sgd'].")

    #使用优化器的minimize方法对损失函数进行最小化操作，得到一个优化步骤self._learn_step
    self._learn_step = optimizer.minimize(self._loss)
    #从平均策略中采样生成一个完整的策略
    self._sample_episode_policy()

  #定义了一个上下文管理器（context manager）temp_mode_as，用于临时更改模式（mode）的值
  @contextlib.contextmanager
  def temp_mode_as(self, mode):
    """Context manager to temporarily overwrite the mode."""
    previous_mode = self._mode
    self._mode = mode
    yield
    self._mode = previous_mode

  #返回步骤计数器（step counter）self._step_counter的值
  def get_step_counter(self):
    return self._step_counter

  #根据一定的概率采样生成一个策略
  def _sample_episode_policy(self):
    if np.random.rand() < self._anticipatory_param:
      self._mode = MODE.best_response#采样最佳响应策略
    else:
      self._mode = MODE.average_policy#采样平均策略

  #用于根据信息状态（info_state）和合法操作（legal_actions）选择一个动作
  def _act(self, info_state, legal_actions):
    info_state = np.reshape(info_state, [1, -1])
    action_values, action_probs = self._session.run(
        [self._avg_policy, self._avg_policy_probs],
        feed_dict={self._info_state_ph: info_state})

    self._last_action_values = action_values[0]
    # Remove illegal actions, normalize probs
    probs = np.zeros(self._num_actions)
    probs[legal_actions] = action_probs[0][legal_actions]
    probs /= sum(probs)
    action = np.random.choice(len(probs), p=probs)
    return action, probs

  #通过访问对象的mode属性来获取self._mode的值
  @property
  def mode(self):
    return self._mode

  #通过访问对象的loss属性来获取(self._last_sl_loss_value, self._last_rl_loss_value())的值
  @property
  def loss(self):
    return (self._last_sl_loss_value, self._last_rl_loss_value())

  #执行一个时间步骤的操作，并返回一个rl_agent.StepOutput对象，包含动作概率和选择的动作
  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the Q-networks if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    #print('mode:',self._mode)
    if self._mode == MODE.best_response: #用DQNagent
      agent_output = self._rl_agent.step(time_step, is_evaluation)
      if not is_evaluation and not time_step.last():
        self._add_transition(time_step, agent_output)

    elif self._mode == MODE.average_policy: #用rl_agent
      # Act step: don't act at terminal info states.
      if not time_step.last():
        #info_state = time_step.observations["info_state"][self.player_id]
        #legal_actions = time_step.observations["legal_actions"][self.player_id]
        current_player = time_step.observations['current_player']
        info_state = time_step.observations["info_state"][current_player]
        legal_actions = time_step.observations["legal_actions"][current_player]
        # print('current_player:',current_player)
        # print('info_state:',info_state)
        # print('legal_actions',legal_actions)
        action, probs = self._act(info_state, legal_actions)
        agent_output = rl_agent.StepOutput(action=action, probs=probs)

      if self._prev_timestep and not is_evaluation:
        self._rl_agent.add_transition(self._prev_timestep, self._prev_action,
                                      time_step)
    else:
      raise ValueError("Invalid mode ({})".format(self._mode))

    if not is_evaluation:
      #增加步骤计数器self._step_counter的值
      self._step_counter += 1

      #如果步骤计数器可以被self._learn_every整除，调用self._learn()方法更新网络并返回最后的损失值
      if self._step_counter % self._learn_every == 0:
        self._last_sl_loss_value = self._learn()
        # If learn step not triggered by rl policy, learn.
        if self._mode == MODE.average_policy:
          self._rl_agent.learn()#调用self._rl_agent.learn()方法进行增强学习

      # Prepare for the next episode.
      #如果时间步骤为最后一个时间步骤，调用self._sample_episode_policy()方法以采样下一episode的策略
      if time_step.last():
        self._sample_episode_policy()
        self._prev_timestep = None
        self._prev_action = None
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = agent_output.action

    return agent_output

  #将新的转换（transition）添加到缓冲区（reservoir buffer）中
  def _add_transition(self, time_step, agent_output):
    """Adds the new transition using `time_step` to the reservoir buffer.

    Transitions are in the form (time_step, agent_output.probs, legal_mask).

    Args:
      time_step: an instance of rl_environment.TimeStep.
      agent_output: an instance of rl_agent.StepOutput.
    """
    legal_actions = time_step.observations["legal_actions"][self.player_id]
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(time_step.observations["info_state"][self.player_id][:]),
        action_probs=agent_output.probs,
        legal_actions_mask=legal_actions_mask)
    self._reservoir_buffer.add(transition)

  #计算采样转换的损失，并执行平均网络的更新操作
  def _learn(self):
    """Compute the loss on sampled transitions and perform a avg-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """
    if (len(self._reservoir_buffer) < self._batch_size or
        len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._reservoir_buffer.sample(self._batch_size)
    info_states = [t.info_state for t in transitions]
    action_probs = [t.action_probs for t in transitions]
    legal_actions_mask = [t.legal_actions_mask for t in transitions]

    loss, _ = self._session.run(
        [self._loss, self._learn_step],
        feed_dict={
            self._info_state_ph: info_states,
            self._action_probs_ph: action_probs,
            self._legal_actions_mask_ph: legal_actions_mask,
        })
    return loss

  #用于生成完整的检查点文件名
  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return os.path.join(checkpoint_dir, checkpoint_filename)
  
  #用于生成最新的检查点文件名
  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return checkpoint_filename + "_latest"

  #用于保存average policy network和内部强化学习代理的Q网络
  def save(self, checkpoint_dir):
    """Saves the average policy network and the inner RL agent's q-network.

    Note that this does not save the experience replay buffers and should
    only be used to restore the agent's policy, not resume training.

    Args:
      checkpoint_dir: directory where checkpoints will be saved.
    """
    for name, saver in self._savers:
      path = saver.save(
          self._session,
          self._full_checkpoint_name(checkpoint_dir, name),
          latest_filename=self._latest_checkpoint_filename(name))
      logging.info("Saved to path: %s", path)

  #用于检查指定目录下是否存在检查点文件
  def has_checkpoint(self, checkpoint_dir):
    for name, _ in self._savers:
      if tf.train.latest_checkpoint(
          self._full_checkpoint_name(checkpoint_dir, name),
          os.path.join(checkpoint_dir,
                       self._latest_checkpoint_filename(name))) is None:
        return False
    return True

  #用于恢复平均策略网络和内部强化学习代理的Q网络
  def restore(self, checkpoint_dir):
    """Restores the average policy network and the inner RL agent's q-network.

    Note that this does not restore the experience replay buffers and should
    only be used to restore the agent's policy, not resume training.

    Args:
      checkpoint_dir: directory from which checkpoints will be restored.
    """
    for name, saver in self._savers:
      full_checkpoint_dir = self._full_checkpoint_name(checkpoint_dir, name)
      logging.info("Restoring checkpoint: %s", full_checkpoint_dir)
      saver.restore(self._session, full_checkpoint_dir)


class ReservoirBuffer(object):
  """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.

  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

  def __init__(self, reservoir_buffer_capacity):
    self._reservoir_buffer_capacity = reservoir_buffer_capacity
    self._data = []
    self._add_calls = 0

  def add(self, element):
    """Potentially adds `element` to the reservoir buffer.

    Args:
      element: data to be added to the reservoir buffer.
    """
    if len(self._data) < self._reservoir_buffer_capacity:
      self._data.append(element)
    else:
      idx = np.random.randint(0, self._add_calls + 1)
      if idx < self._reservoir_buffer_capacity:
        self._data[idx] = element
    self._add_calls += 1

  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.

    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
          num_samples, len(self._data)))
    return random.sample(self._data, num_samples)

  def clear(self):
    self._data = []
    self._add_calls = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)
