import collections
import contextlib
import enum
import os
import random
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import itertools

from open_spiel.python import simple_nets

tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition", "info_state real_cards_probs legal_cards_mask")

class card_predict(object):
  """implementation in TensorFlow.
     predict hand card for the player's teammate. 
  """
  def __init__(self,
               session,
               state_representation_size,
               num_dices,
               num_dice_sides,
               num_players,
               hidden_layers_sizes,
               reservoir_buffer_capacity,
               batch_size=128,
               sl_learning_rate=0.002,
               min_buffer_size_to_learn=2000,
               learn_every=64,
               optimizer_str="adam"):
    """Initialize the card predict agent."""
    self._session = session
    self._num_players = num_players
    self._num_teammates = self._num_players - 2
    self._num_dices = num_dices
    self._num_dice_sides = num_dice_sides
    self._num_cards = num_dice_sides**num_dices
    self._output_cards_list = self.generate_all_possible_cards(self._num_cards, self._num_teammates)
    self._output_length = len(self._output_cards_list)
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._learn_every = learn_every
    self._min_buffer_size_to_learn = min_buffer_size_to_learn

    self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning.
    self._step_counter = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_sl_loss_value = None

    # Placeholders.
    self._info_state_ph = tf.placeholder(
        shape=[None, state_representation_size],
        dtype=tf.float32,
        name="info_state_ph")

    self._cards_probs_ph = tf.placeholder(
        shape=[None, self._output_length], dtype=tf.float32, name="cards_probs_ph")

    self._legal_cards_mask_ph = tf.placeholder(
        shape=[None, self._output_length],
        dtype=tf.float32,
        name="legal_cards_mask_ph")

    # Card predict network.
    self._card_predict_network = simple_nets.MLP(state_representation_size,
                                        self._layer_sizes, self._output_length)
    self._cards_logits = self._card_predict_network(self._info_state_ph)
    self._cards_probs = tf.nn.softmax(self._cards_logits)

    self._savers = [
        ("card_predict_network", tf.train.Saver(self._card_predict_network.variables))
    ]

    # Loss
    self._loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(self._cards_probs_ph),
            logits=self._cards_logits))

    if optimizer_str == "adam":
      optimizer = tf.train.AdamOptimizer(learning_rate=sl_learning_rate)
    elif optimizer_str == "sgd":
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=sl_learning_rate)
    else:
      raise ValueError("Not implemented. Choose from ['adam', 'sgd'].")

    self._learn_step = optimizer.minimize(self._loss)

  # 生成队友手牌的所有可能排列情况
  def generate_all_possible_cards(self, cards, num_teammates):
    cards_list = [i for i in range(cards)]
    permutations = list(itertools.product(cards_list, r=num_teammates))
    permutations_as_lists = [list(perm) for perm in permutations]
    return permutations_as_lists
  
  def _predict(self, info_state, legal_cards):
    info_state = np.reshape(info_state, [1, -1])
    cards_values, cards_probs = self._session.run(
        [self._cards_logits, self._cards_probs],
        feed_dict={self._info_state_ph: info_state})

    self._last_cards_values = cards_values[0]
    # Remove illegal cards, normalize probs
    probs = np.zeros(self._output_length)
    probs[legal_cards] = cards_probs[0][legal_cards]
    probs /= sum(probs)

    # card 由随机选取改为选取概率最大的
    # card = np.random.choice(len(probs), p=probs)
    cards_index = np.argmax(probs)
    cards = self._output_cards_list[cards_index]
    return cards_index, probs, cards

 
  def loss(self):
    return (self._last_sl_loss_value)


  def step(self, time_step, is_evaluation=False):
    """Returns the predicted card.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    """

    info_state = self.get_imperfect_info_state(time_step)
    real_card = self.get_real_teammate_card(time_step)
    legal_cards = self.get_legal_cards(time_step)
    # card, probs = self._predict(info_state, legal_actions)

    self._add_transition(info_state, real_card, legal_cards)

    if not is_evaluation:
      self._step_counter += 1

      if self._step_counter % self._learn_every == 0:
        self._last_sl_loss_value = self._learn()
        # print("step: ", self._step_counter," loss: ", self._last_sl_loss_value)


  def get_imperfect_info_state(self, time_step):
    """ return  imperfect_info_state. """
    current_player = time_step.observations["current_player"]
    origin_info_state = time_step.observations["info_state"][current_player]
    imperfect_info_state = []
    dice_bits_for_each_player = self._num_dices * self._num_dice_sides
    start = self._num_players + dice_bits_for_each_player * current_player
    end = self._num_players + dice_bits_for_each_player * current_player + dice_bits_for_each_player -1
    for i in range(len(origin_info_state)):
      if i < self._num_players:
        imperfect_info_state.append(origin_info_state[i])
      # num_players+num_cards*current_player 
      elif i >=  start and i <= end:
        imperfect_info_state.append(origin_info_state[i])
      elif i>= self._num_players + dice_bits_for_each_player * self._num_players:
        imperfect_info_state.append(origin_info_state[i])
    return imperfect_info_state

  
  
  def get_legal_cards(self, time_step):
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
    

  def get_real_teammate_card(self, time_step):
    """ return the teammate's real hand cards' index in the output_cards_list. """
    current_player = time_step.observations["current_player"]
    origin_info_state = time_step.observations["info_state"][current_player]
    dice_bits_for_each_player = self._num_dices * self._num_dice_sides

    start = self._num_players 
    end = self._num_players + dice_bits_for_each_player * self._num_players
    # 将玩家手牌部分向量截取出来并分割成每个玩家的手牌切片
    allPlayers_card_list = origin_info_state[start:end]
    teammates_card_list =  [allPlayers_card_list[i:i+dice_bits_for_each_player] for i in range(0, len(allPlayers_card_list), dice_bits_for_each_player)]
    
    real_cards = []
    # 编码方式为：在2个骰子三面的情况下，若队友玩家手牌为[010 001]即32(第一组三个数为个位，第二组三个数为十位，以此类推)，则对应编码为7，计算方式为（3-1)*3^1 + (2-1)*3^0
    for i in range(1, self._num_players):
      if i != current_player:
        # 提取每个玩家的手牌片段
        tmp = teammates_card_list[i]
        tmp_list =  [tmp[i:i+self._num_dice_sides] for i in range(0, len(tmp), self._num_dice_sides)]
        real_card = 0
        for x in range(self._num_dices):
          tmp_bit = next(j for j, card in enumerate(tmp_list[x]) if card) + 1
          real_card += (tmp_bit-1)*(self._num_dice_sides**(x))    
        real_cards.append(real_card)
    
    
    # 寻找队友手牌集合对应在所有可能手牌组合list中的索引并返回该索引
    real_cards_index = next(i for i, sublist in enumerate(self._output_cards_list) if sublist == real_cards)
    return real_cards_index 
    

  def _add_transition(self, info_state, real_card, legal_cards):
    """Adds the new transition to the reservoir buffer.

    Transitions are in the form (info_state, real_cards_probs, legal_cards).

    Args:
      info_state: an imperfect infostate of the current team player.
      real_card: the teammate's real hand card.
      legal_cards: a list of all the possible cards the teammate may hold.
    """
    real_cards_probs = np.zeros(self._output_length)
    real_cards_probs[real_card] = 1.0
    legal_cards_mask = np.zeros(self._output_length)
    legal_cards_mask[legal_cards] = 1.0
    transition = Transition(
        info_state=info_state,
        real_cards_probs=real_cards_probs,
        legal_cards_mask = legal_cards_mask)
    self._reservoir_buffer.add(transition)


  def _learn(self):
    """Compute the loss on sampled transitions and perform a card_predict_network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The loss obtained on this batch of transitions or `None`.
    """
    if (len(self._reservoir_buffer) < self._batch_size or
        len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._reservoir_buffer.sample(self._batch_size)
    info_states = [t.info_state for t in transitions]
    cards_probs = [t.real_cards_probs for t in transitions]
    legal_cards_masks = [t.legal_cards_mask for t in transitions]

    loss, _ = self._session.run(
        [self._loss, self._learn_step],
        feed_dict={
            self._info_state_ph: info_states,
            self._cards_probs_ph: cards_probs,
            self._legal_cards_mask_ph: legal_cards_masks,
        })
    return loss
  
  def _compute_accuracy(self):
    """Compute the accuracy on sampled transitions .

    If there are not enough elements in the buffer, no acuuracy is computed and
    `None` is returned instead.

    Returns:
      The predict accuracy on this batch of transitions or `None`.
    """
    sample_size = 2000
    if (len(self._reservoir_buffer) < sample_size):
      return None
    
    transitions = self._reservoir_buffer.sample(sample_size)
    info_states = [t.info_state for t in transitions]
    cards_probs = [t.real_cards_probs for t in transitions]
    legal_cards_masks = [t.legal_cards_mask for t in transitions]

    real_cards = [[index for index, value in enumerate(inner_lst) if value != 0] for inner_lst in cards_probs]
    legal_cards_list = [[index for index, value in enumerate(inner_lst) if value != 0] for inner_lst in legal_cards_masks]
    
    sum = 0
    accurate_sum = 0
    
    for info_state, legal_cards, real_card in zip(info_states, legal_cards_list, real_cards):
      card, _, _ = self._predict(info_state, legal_cards)
      sum += 1
      if card == real_card:
        accurate_sum += 1
    
    return accurate_sum/sum

  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = name
    return os.path.join(checkpoint_dir, checkpoint_filename)

  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = name
    return checkpoint_filename + "_latest"

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

  def has_checkpoint(self, checkpoint_dir):
    for name, _ in self._savers:
      if tf.train.latest_checkpoint(
          self._full_checkpoint_name(checkpoint_dir, name),
          os.path.join(checkpoint_dir,
                       self._latest_checkpoint_filename(name))) is None:
        return False
    return True

  def restore(self, checkpoint_dir):
    """Restores the predict network.

    Note that this does not restore the experience replay buffers and should
    only be used to restore the predict network, not resume training.

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

