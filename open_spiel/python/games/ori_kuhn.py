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

# Lint as python3
"""Kuhn Poker implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum

import numpy as np

import pyspiel

#在游戏中可以采取的动作
class Action(enum.IntEnum):
  PASS = 0
  BET = 1

#游戏中的玩家数量、真实玩家数量和牌的数量
_NUM_PLAYERS = 2
_REAL_PLAYERS = 3
_RANK = 6

#创建了一个包含了所有牌的牌堆
_DECK = set()
for i in range(0, _RANK):
  _DECK.add(i)

#定义了一个名为_GAME_TYPE的pyspiel.GameType对象#
#表示游戏的类型和特性。这个对象包含了有关游戏的信息，例如游戏的名称、动态、机会模式、信息性质、效用类型等
_GAME_TYPE = pyspiel.GameType(
    short_name="ori_kuhn",
    long_name="Original Kuhn Poker",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=True)

#定义了一个名为_GAME_INFO的pyspiel.GameInfo对象
#包含了游戏的信息，例如可用的动作数量、最大机会结果数量、玩家数量、效用范围等。
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(Action),
    max_chance_outcomes=len(_DECK),
    num_players=_NUM_PLAYERS,
    min_utility=-(_REAL_PLAYERS - 1)*2,
    max_utility=(_REAL_PLAYERS - 1)*2,
    utility_sum=0.0,
    max_game_length=_REAL_PLAYERS*2-1)  # e.g. Pass, Bet, Bet

#表示Kuhn扑克游戏本身，包含了游戏的规则和动态
class KuhnPokerGame(pyspiel.Game):
  """A Python version of Kuhn poker."""

  #初始化游戏对象
  def __init__(self, params=None):
    print("This is original kuhn_poker!!")
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  #用于创建游戏的初始状态
  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return KuhnPokerState(self)

  #创建用于观察游戏状态的观察者对象
  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return KuhnPokerObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)

#表示Kuhn扑克游戏的状态
class KuhnPokerState(pyspiel.State):
  """A python version of the Kuhn poker state."""

  #初始化了游戏状态的各种属性，包括玩家的牌、赌注、奖池、游戏是否结束以及下一位玩家
  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.cards = []
    self.bets = []
    self.pot = [1.0] * _REAL_PLAYERS
    self._game_over = False
    self._next_player = 0

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  #返回当前轮到移动的玩家的ID
  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif len(self.cards) < _REAL_PLAYERS:
      return pyspiel.PlayerId.CHANCE
    elif self._next_player == 0:
      return 0
    else:
      return 1

  #返回指定玩家在当前状态下可以采取的合法动作列表
  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    return [Action.PASS, Action.BET]

  #在机会节点（发牌阶段）时返回可能的机会结果和它们的概率
  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    outcomes = sorted(_DECK - set(self.cards))
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]

  #将指定的动作应用到游戏状态上
  def _apply_action(self, action):
    """Applies the specified action to the state."""
    #如果当前状态是机会节点，表示发牌
    if self.is_chance_node():
      self.cards.append(action)
    else:
      #表示下注或跟注
      self.bets.append(action)
      #更新奖池中相应玩家的奖池值
      if action == Action.BET:
        self.pot[self._next_player] += 1
      self._next_player = (self._next_player + 1) % _REAL_PLAYERS
      #检查游戏是否结束
      if ((len(self.bets) == _REAL_PLAYERS and len(set(self.bets)) == 1) or  #bbb或者ppp
          (len(self.bets) >= _REAL_PLAYERS and self.bets[len(self.bets) - _REAL_PLAYERS] == 1)): #其他情况，看第一轮该选手的上一位是否选择了b即可
        self._game_over = True

  #将动作转换为字符串，以便进行输出或显示
  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal:{action}"
    elif action == Action.PASS:
      return "Pass"
    else:
      return "Bet"

  #返回一个布尔值，指示游戏是否已经结束
  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  #用于计算游戏结束时各个玩家的总奖励
  def returns(self):
    """Total reward for each player over the course of the game so far."""
    pot = self.pot
    beter_index = []
    winnings = 0.0
    for i in range(1, _REAL_PLAYERS):
      winnings += pot[i]
    if not self._game_over:
      return [0., 0.]
    elif pot[0] == 1.0 and len(set(pot)) != 1:  # 对手没有下注，但团队有人下注
      return [-pot[0], pot[0]]
    elif pot[0] == 1.0 and max(pot) == 1.0:  # 对手和团队玩家都没有下注
      for i in range(1, _REAL_PLAYERS):
        if self.cards[0] < self.cards[i]:
          return [-pot[0], pot[0]]
      return [winnings, -winnings]
    else:  #对手有下注
      for beter in range(_REAL_PLAYERS): # 找到有下注的人
        if pot[beter] == 2.0:
          beter_index.append(beter)
      for i in beter_index:  # 对手只要输给一个团队成员就算输
        if self.cards[0] < self.cards[i]:
          return [-pot[0], pot[0]]
      return [winnings, -winnings]
  
  #用于计算奖励分配的潜在奖池
  def get_return_pot(self, winnings):
    result = []
    result.append(winnings)
    for p in range(1, _REAL_PLAYERS):
        result.append(-winnings/(_REAL_PLAYERS-1))
    return result

  #用于将游戏状态转化为可打印的字符串
  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return "".join([str(c) for c in self.cards] + ["pb"[b] for b in self.bets])



#用于观察游戏状态并生成观察信息
#实现了OpenSpiel库中的观察者接口，以便可以将游戏状态转换为适用于算法和智能体的观察信息
class KuhnPokerObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  #初始化观察者对象
  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Determine which observation pieces we want to include.
    #首先根据iig_obs_type的设置来确定要包含的观察信息组件（pieces）。这些组件包括玩家、私有牌、公共信息等
    pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))]
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("private_card", _RANK, (_RANK, )))
    if iig_obs_type.public_info:
      if iig_obs_type.perfect_recall:
        game_length = _NUM_PLAYERS * 2 - 1
        pieces.append(("betting", game_length*_NUM_PLAYERS, (game_length, _NUM_PLAYERS)))
      else:
        pieces.append(("pot_contribution", _NUM_PLAYERS, (_NUM_PLAYERS,)))

    # Build the single flat tensor.
    #总大小为total_size的一维数组self.tensor，用于存储观察信息
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    #构建一个名为self.dict的字典，用于存储观察信息的命名和重塑后的视图。
    #这个字典将每个观察信息组件与其对应的部分（子数组）相关联。
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size
  #用于根据游戏状态state和指定的玩家player更新观察信息
  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    # print("state: ", str(state), "----cards: ", state.cards, "----player: ", player)
    if "player" in self.dict:
      self.dict["player"][player] = 1
    if "private_card" in self.dict and len(state.cards) > player:
      self.dict["private_card"][state.cards[player]] = 1
    if "pot_contribution" in self.dict:
      self.dict["pot_contribution"][:] = state.pot
    if "betting" in self.dict:
      for turn, action in enumerate(state.bets):
        self.dict["betting"][turn, action] = 1
        
  #用于将观察信息以字符串形式表示
  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    trans_player = state._next_player # 实际上的玩家编号
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "private_card" in self.dict and len(state.cards) > player:
      pieces.append(f"card:{state.cards[trans_player]}")
    if "pot_contribution" in self.dict:
      pieces.append(f"pot[{int(state.pot[i])} {int(state.pot[i])}]")
    if "betting" in self.dict and state.bets:
      pieces.append("".join("pb"[b] for b in state.bets))
    return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, KuhnPokerGame)
