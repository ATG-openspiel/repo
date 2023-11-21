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
"""Leduc Poker implemented in Python.

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


class Action(enum.IntEnum):
  FOLD = 0
  CALL = 1
  RAISE = 2 


_NUM_PLAYERS = 2
_REAL_PLAYERS = 3
_FIRST_RAISE_AMOUNT = 2
_SECOND_RAISE_AMOUNT = 4
_MAX_RAISE_PER_ROUND = 2
_RANK = 3
_SUIT = 3
_DECK = set()
for i in range(_RANK):
  _DECK.add(i)
_GAME_TYPE = pyspiel.GameType(
    short_name="ori_leduc",
    long_name="Original Leduc Poker",
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
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(Action),
    max_chance_outcomes=len(_DECK),
    num_players=_NUM_PLAYERS,
    min_utility=-1 * (_MAX_RAISE_PER_ROUND * _FIRST_RAISE_AMOUNT + _MAX_RAISE_PER_ROUND * _SECOND_RAISE_AMOUNT + 1),
    max_utility=(_REAL_PLAYERS - 1) * (_MAX_RAISE_PER_ROUND * _FIRST_RAISE_AMOUNT + _MAX_RAISE_PER_ROUND * _SECOND_RAISE_AMOUNT + 1),
    utility_sum=0.0,
    max_game_length=2 * (3 * _REAL_PLAYERS - 2))  # e.g. Pass, Bet, Bet


class LeducPokerGame(pyspiel.Game):
  """A Python version of Leduc poker."""

  def __init__(self, params=None):
    print("This is original leduc_poker!!")
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return LeducPokerState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return LeducPokerObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class LeducPokerState(pyspiel.State):
  """A python version of the Leduc poker state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.cards = []
    self.public_card = -1
    self.bets = []
    self.first_round_bets = []
    self.second_round_bets = []
    self.pot = [1.0] * _REAL_PLAYERS
    self._game_over = False
    self._next_player = 0
    self.round = 1
    self.num_raise = 0
    self.num_call = 0
    self.stake = 1 # 当前这一轮的最大注
    self.remaining_players = _REAL_PLAYERS
    self.folded = [False] * _REAL_PLAYERS
    self.invalid_action = []

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif (self.round == 1 and len(self.cards) < _REAL_PLAYERS) or (self.round == 2 and self.public_card == -1):
      return pyspiel.PlayerId.CHANCE
    elif self._next_player == 0:
      return 0
    else:
      return 1

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    legal_action_list = []
    if max(self.pot) > self.pot[self._next_player]:
      legal_action_list.append(Action.FOLD)
    legal_action_list.append(Action.CALL)
    if self.num_raise < _MAX_RAISE_PER_ROUND:
      legal_action_list.append(Action.RAISE)
    return legal_action_list

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    outcomes = sorted(_DECK)
    for i in range(_RANK):
      if self.cards.count(i) == _SUIT:
        outcomes = sorted(_DECK - {i})
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      if len(self.cards) < _REAL_PLAYERS: # round 1:deal cards to each player
        self.cards.append(action)
      else: # round 2:deal a single public card
        self.public_card = action
    else:
      self.bets.append(action)
      if self.round == 1:
        self.first_round_bets.append(action)
      else:
        self.second_round_bets.append(action)
      if action == Action.FOLD:
        self.folded[self._next_player] = True
        self.remaining_players -= 1
        if self._terminal():
          self._game_over = True
        elif self._ready_for_next_round():
          self._newround()
        else:
          self._next_player = self._nextplayer()
      elif action == Action.CALL:
        amount = self.stake - self.pot[self._next_player]
        self.pot[self._next_player] += amount
        self.num_call += 1
        if self._terminal():
          self._game_over = True
        elif self._ready_for_next_round():
          self._newround()
        else:
          self._next_player = self._nextplayer()
      elif action == Action.RAISE:
        call_amount = self.stake - self.pot[self._next_player]
        if call_amount > 0:
          self.pot[self._next_player] += call_amount
        raise_amount = _FIRST_RAISE_AMOUNT if self.round == 1 else _SECOND_RAISE_AMOUNT
        self.stake += raise_amount
        self.pot[self._next_player] += raise_amount
        self.num_raise += 1
        self.num_call = 0
        if self._terminal():
          print("can the game terminal after someone raise?")
          self._game_over = True
        else:
          self._next_player = self._nextplayer()
      else:
        print("The action is invalid")

      
  # 类内部使用，判断是否游戏结束
  def _terminal(self):
    return self.remaining_players == 1 or (self.round == 2 and self._ready_for_next_round())
  
  
  def _ready_for_next_round(self):
    return (self.num_raise == 0 and self.num_call == self.remaining_players) or (self.num_raise > 0 and self.num_call == (self.remaining_players - 1))
  
  def _newround(self):
    assert self.round == 1
    self.round += 1
    self.num_call = 0
    self.num_raise = 0
    for i in range(_REAL_PLAYERS):
      if self.folded[i] is not True:
        self._next_player = i
        return

  def _nextplayer(self):
    current_real_player = self._next_player
    for i in range(1, _REAL_PLAYERS):
      player = (current_real_player + i) % _REAL_PLAYERS
      if self.folded[player] is not True:
        return player

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal:{action}"
    elif action == Action.FOLD:
      return "Fold"
    elif action == Action.CALL:
      return "Call"
    else:
      return "Raise"
  
  # 可外部使用
  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    pot = self.pot
    winning = 0.0
    handrank_list = []
    for player in range(_REAL_PLAYERS):
      if self.folded[player] is False:
        handrank_list.append(self._handrank(player))
    max_rank = max(handrank_list)
    if self.folded[0] is True:  # 对手弃牌
      return [-pot[0], pot[0]]
    elif self._handrank(0) != max_rank: # 对手没弃牌但是输了
      return [-pot[0], pot[0]]
    else:  #对手没弃牌，有可能赢或者打平
      for i in range(1, len(handrank_list)):
        if handrank_list[i] == handrank_list[0]:
          return [0.0] * _NUM_PLAYERS   # 这里是打平了
      # 这里对手赢了
      for i in range(1, _REAL_PLAYERS):
        winning += pot[i]
      return [winning, -winning]


  def _handrank(self, player):
    hand = [self.cards[player], self.public_card]
    num_cards = len(_DECK)
    if hand[0] == hand[1]:
      return num_cards * num_cards
    else:
      return hand[0] * num_cards
    

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return "".join([str(c) for c in self.cards] + list(str(self.public_card)) + ["fcr"[b] for b in self.bets])


class LeducPokerObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Determine which observation pieces we want to include.
    pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))]
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("private_card", _RANK, (_RANK, )))
    if iig_obs_type.public_info:
      if iig_obs_type.perfect_recall:
        game_length = 2 * (3 * _NUM_PLAYERS - 2)
        pieces.append(("betting", game_length*_NUM_PLAYERS, (game_length, _NUM_PLAYERS)))
        pieces.append(("public_card", 1, (1,)))
      else:
        pieces.append(("pot_contribution", _NUM_PLAYERS, (_NUM_PLAYERS,)))

    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

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

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    trans_player = state._next_player # 实际上的玩家编号
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "private_card" in self.dict and len(state.cards) > trans_player:
      pieces.append(f"card:{state.cards[trans_player]}")
    if "pot_contribution" in self.dict:
      pieces.append(f"pot[{int(state.pot[i])} {int(state.pot[i])}]")
    if "betting" in self.dict and state.bets:
      if state.round == 1:
        pieces.append("".join("fcr"[b] for b in state.first_round_bets))
      if "public_card" in self.dict and state.round == 2:
        pieces.append("".join(["fcr"[b] for b in state.first_round_bets] + 
                              list(str(state.public_card)) + 
                              ["fcr"[b] for b in state.second_round_bets]))
      
    return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, LeducPokerGame)
