// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


//这是对抗性团队博弈版本leduc
//leduc for ATG

#include "open_spiel/games/leduc_mp_full/leduc_mp_full.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace leduc_mp_full {
namespace {

constexpr double kAnte = 1;//盲注

const GameType kGameType{/*short_name=*/"leduc_mp_full",
                         /*long_name=*/"Leduc Poker mp full",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/10,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultPlayers)},
                          {"action_mapping", GameParameter(false)},
                          {"suit_isomorphism", GameParameter(false)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LeducGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string StatelessActionToString(Action action) {
  if (action == ActionType::kFold) {
    return "Fold";
  } else if (action == ActionType::kCall) {
    return "Call";
  } else if (action == ActionType::kRaise) {
    return "Raise";
  } else {
    SpielFatalError(absl::StrCat("Unknown action: ", action));
    return "Will not return.";
  }
}

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

// The Observer class is responsible for creating representations of the game
// state for use in learning algorithms. It handles both string and tensor
// representations, and any combination of public information and private
// information (none, observing player only, or all players).
//
// If a perfect recall observation is requested, it must be possible to deduce
// all previous observations for the same information type from the current
// observation.

class LeducObserver : public Observer {
 public:
  LeducObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  //
  // These helper methods each write a piece of the tensor observation.
  //

  // Identity of the observing player. One-hot vector of size num_players.
  static void WriteObservingPlayer(const LeducState& state, int player,
                                   Allocator* allocator) {
    auto out = allocator->Get("player", {state.num_players_});
    out.at(player) = 1;
  }

  // Private card of the observing player. One-hot vector of size num_cards.
  // static void WriteSinglePlayerCard(const LeducState& state, int player,
  //                                   Allocator* allocator) {
  //   auto out = allocator->Get("private_card", {state.NumObservableCards()});
  //   int card = state.private_cards_[player];
  //   if (card != kInvalidCard) out.at(card) = 1;
  // }

  static void WriteSinglePlayerCard(const LeducState& state, int player,
                                    Allocator* allocator) {
    auto out = allocator->Get("private_card", {state.num_players_, state.NumObservableCards()});
    if(player == 0){// 当前观察者是player0，只标识player0的牌
      int card = state.private_cards_[player];
      if (card != kInvalidCard) out.at(player, card) = 1;
    }else{
      for (int p = 0; p < state.num_players_; ++p) {
        if(p!=0){
          int card = state.private_cards_[p];
          if (card != kInvalidCard) out.at(p, state.private_cards_[p]) = 1;
        }
    }
    }
    
  }

  // Private cards of all players. Tensor of shape [num_players, num_cards].
  static void WriteAllPlayerCards(const LeducState& state,
                                  Allocator* allocator) {
    auto out = allocator->Get("private_cards",
                              {state.num_players_, state.NumObservableCards()});
    for (int p = 0; p < state.num_players_; ++p) {
      int card = state.private_cards_[p];
      if (card != kInvalidCard) out.at(p, state.private_cards_[p]) = 1;
    }
  }

  // Community card (if any). One-hot vector of size num_cards.
  static void WriteCommunityCard(const LeducState& state,
                                 Allocator* allocator) {
    auto out = allocator->Get("community_card", {state.NumObservableCards()});
    if (state.public_card_ != kInvalidCard) {
      out.at(state.public_card_) = 1;
    }
  }

  // Betting sequence; shape [num_rounds, bets_per_round, num_actions].
  static void WriteBettingSequence(const LeducState& state,
                                   Allocator* allocator) {
    const int kNumRounds = 2;
    const int kBitsPerAction = 2;
    const int max_bets_per_round = state.MaxBetsPerRound();
    // std::cout<<max_bets_per_round<<"--MaxBets"<<std::endl;
    auto out = allocator->Get("betting",
                              {kNumRounds, max_bets_per_round, kBitsPerAction});
    for (int round : {0, 1}) {
      const auto& bets =
          (round == 0) ? state.round1_sequence_ : state.round2_sequence_;
      for (int i = 0; i < bets.size(); ++i) {
        if (bets[i] == ActionType::kCall) {
          out.at(round, i, 0) = 1;  // Encode call as 10. //bug
        } else if (bets[i] == ActionType::kRaise) {
          out.at(round, i, 1) = 1;  // Encode raise as 01.
        }
      }
    }
  }

  // Pot contribution per player (integer per player).
  static void WritePotContribution(const LeducState& state,
                                   Allocator* allocator) {
    auto out = allocator->Get("pot_contribution", {state.num_players_});
    for (auto p = Player{0}; p < state.num_players_; p++) {
      out.at(p) = state.ante_[p];
    }
  }

  // Writes the complete observation in tensor form.
  // The supplied allocator is responsible for providing memory to write the
  // observation into.
  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    auto& state = open_spiel::down_cast<const LeducState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    // Observing player.
    WriteObservingPlayer(state, player, allocator);
    // Private card(s).
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      WriteSinglePlayerCard(state, player, allocator);
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      WriteAllPlayerCards(state, allocator);
    }
    // Public information.
    if (iig_obs_type_.public_info) {
      WriteCommunityCard(state, allocator);
      //bug WriteBettingSequence(state, allocator)
      iig_obs_type_.perfect_recall ? WriteBettingSequence(state, allocator)
                                   : WritePotContribution(state, allocator);
      
    }
  }

  // Writes an observation in string form. It would be possible just to
  // turn the tensor observation into a string, but we prefer something
  // somewhat human-readable.

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    auto& state = open_spiel::down_cast<const LeducState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);
    std::string result;

    // Private card(s).
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      absl::StrAppend(&result, "[Observer: ", player, "]");
      //absl::StrAppend(&result, "[Private: ", state.private_cards_[player], "]");
      if(player == 0){
        absl::StrAppend(&result, "[Private: ", state.private_cards_[player], "]");
      }else{
        absl::StrAppend(&result, "[Privates: ");
        absl::StrAppend(&result, absl::StrJoin(state.private_cards_.begin() + 1, state.private_cards_.end(), ""));
        absl::StrAppend(&result, "]");
      }
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      absl::StrAppend(
          &result, "[Privates: ", absl::StrJoin(state.private_cards_, ""), "]");
    }

    // Public info. Not all of this is strictly necessary, but it makes the
    // string easier to understand.
    if (iig_obs_type_.public_info) {
      absl::StrAppend(&result, "[Round ", state.round_, "]");
      absl::StrAppend(&result, "[Player: ", state.cur_player_, "]");
      absl::StrAppend(&result, "[Pot: ", state.pot_, "]");
      absl::StrAppend(&result, "[Money: ", absl::StrJoin(state.money_, " "),
                      "]");
      if (state.public_card_ != kInvalidCard) {
        absl::StrAppend(&result, "[Public: ", state.public_card_, "]");
      }
      if (iig_obs_type_.perfect_recall) {
        // Betting Sequence (for the perfect recall case)
        absl::StrAppend(
            &result, "[Round1: ", absl::StrJoin(state.round1_sequence_, " "),
            "][Round2: ", absl::StrJoin(state.round2_sequence_, " "), "]");
      } else {
        // Pot contributions (imperfect recall)
        absl::StrAppend(&result, "[Ante: ", absl::StrJoin(state.ante_, " "),
                        "]");
      }
    }

    // Done.
    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

LeducState::LeducState(std::shared_ptr<const Game> game, bool action_mapping,
                       bool suit_isomorphism)
    : State(game),
      cur_player_(kChancePlayerId),
      num_calls_(0),
      num_raises_(0),
      round_(1),   // Round number (1 or 2).
      stakes_(1),  // The current 'level' of the bet.
      num_winners_(-1),
      pot_(kAnte * game->NumPlayers()),  // Number of chips in the pot.
      public_card_(kInvalidCard),
      // Number of cards remaining; not equal deck_.size()!
      deck_size_((game->NumPlayers() + kNumRanks) * kNumSuits), //修改rank数量 
      private_cards_dealt_(0),
      remaining_players_(game->NumPlayers()),
      // Is this player a winner? Indexed by pid.
      winner_(game->NumPlayers(), false),
      // Each player's single private card. Indexed by pid.
      private_cards_(game->NumPlayers(), kInvalidCard),
      // How much money each player has, indexed by pid.
      money_(game->NumPlayers(), kStartingMoney - kAnte),
      // How much each player has contributed to the pot, indexed by pid.
      ante_(game->NumPlayers(), kAnte),
      // Flag for whether the player has folded, indexed by pid.
      folded_(game->NumPlayers(), false),
      // Sequence of actions for each round. Needed to report information
      // state.
      round1_sequence_(),
      round2_sequence_(),
      // Always regard all actions as legal, and internally map otherwise
      // illegal actions to check/call.
      action_mapping_(action_mapping),
      // Players cannot distinguish between cards of different suits with the
      // same rank.
      suit_isomorphism_(suit_isomorphism) {
  // Cards by value (0-6 for standard 2-player game, kInvalidCard if no longer
  // in the deck.)
  deck_.resize(deck_size_);
  std::iota(deck_.begin(), deck_.end(), 0);
}

int LeducState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

// In a chance node, `move` should be the card to deal to the current
// underlying player.
// On a player node, it should be ActionType::{kFold, kCall, kRaise}
void LeducState::DoApplyAction(Action move) {
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(move, 0);
    SPIEL_CHECK_LT(move, deck_.size());
    if (suit_isomorphism_) {
      // One of the two identical cards must be left in the deck.
      SPIEL_CHECK_TRUE(deck_[move * 2] != kInvalidCard ||
                       deck_[move * 2 + 1] != kInvalidCard);
    } else {
      SPIEL_CHECK_NE(deck_[move], kInvalidCard);
    }

    if (private_cards_dealt_ < num_players_) {
      SetPrivate(private_cards_dealt_, move);
    } else {
      // Round 2: A single public card.
      if (suit_isomorphism_) {
        public_card_ = move;
        if (deck_[move * 2] != kInvalidCard) {
          deck_[move * 2] = kInvalidCard;
        } else if (deck_[move * 2 + 1] != kInvalidCard) {
          deck_[move * 2 + 1] = kInvalidCard;
        } else {
          SpielFatalError("Suit isomorphism error.");
        }
        deck_size_--;
      } else {
        public_card_ = deck_[move];
        deck_[move] = kInvalidCard;
        deck_size_--;
      }

      // We have finished the public card, let's bet!
      cur_player_ = NextPlayer();
    }
  } else {
    // Player node.
    if (action_mapping_) {
      // Map otherwise illegal actions to kCall.
      if (move == ActionType::kFold) {
        if (stakes_ <= ante_[cur_player_]) {
          move = ActionType::kCall;
        }
      } else if (move == ActionType::kRaise) {
        if (num_raises_ >= kMaxRaises) {//raise次数(times of raise)
          move = ActionType::kCall;
        }
      }
    }

    if (move == ActionType::kFold) {
      SPIEL_CHECK_NE(cur_player_, kChancePlayerId);
      SequenceAppendMove(ActionType::kFold);

      // Player is now out.
      folded_[cur_player_] = true;
      remaining_players_--;

      if (IsTerminal()) {
        ResolveWinner();
      } else if (ReadyForNextRound()) {
        NewRound();
      } else {
        cur_player_ = NextPlayer();
      }
    } else if (move == ActionType::kCall) {
      SPIEL_CHECK_NE(cur_player_, kChancePlayerId);

      // Current player puts in an amount of money equal to the current level
      // (stakes) minus what they have contributed to level their contribution
      // off. Note: this action also acts as a 'check' where the stakes are
      // equal to each player's ante.
      SPIEL_CHECK_GE(stakes_, ante_[cur_player_]);
      int amount = stakes_ - ante_[cur_player_];
      Ante(cur_player_, amount);
      num_calls_++;
      SequenceAppendMove(ActionType::kCall);

      if (IsTerminal()) {
        ResolveWinner();
      } else if (ReadyForNextRound()) {
        NewRound();
      } else {
        cur_player_ = NextPlayer();
      }
    } else if (move == ActionType::kRaise) {
      SPIEL_CHECK_NE(cur_player_, kChancePlayerId);

      // This player matches the current stakes and then brings the stakes up.
      SPIEL_CHECK_LT(num_raises_, kMaxRaises);
      int call_amount = stakes_ - ante_[cur_player_];

      // First, match the current stakes if necessary
      SPIEL_CHECK_GE(call_amount, 0);
      if (call_amount > 0) {
        Ante(cur_player_, call_amount);
      }

      // Now, raise the stakes.
      int raise_amount = (round_ == 1 ? kFirstRaiseAmount : kSecondRaiseAmount);
      stakes_ += raise_amount;
      Ante(cur_player_, raise_amount);
      num_raises_++;
      num_calls_ = 0;
      SequenceAppendMove(ActionType::kRaise);

      if (IsTerminal()) {
        ResolveWinner();
      } else {
        cur_player_ = NextPlayer();
      }
    } else {
      SpielFatalError(absl::StrCat("Move ", move, " is invalid. ChanceNode?",
                                   IsChanceNode()));
    }
  }
}

std::vector<Action> LeducState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> movelist;
  if (IsChanceNode()) {
    if (suit_isomorphism_) {
      // Consecutive cards are identical under suit isomorphism.
      for (int card = 0; card < deck_.size() / 2; card++) {
        if (deck_[card * 2] != kInvalidCard ||
            deck_[card * 2 + 1] != kInvalidCard) {
          movelist.push_back(card);
        }
      }
    } else {
      for (int card = 0; card < deck_.size(); card++) {
        if (deck_[card] != kInvalidCard) movelist.push_back(card);
      }
    }
    return movelist;
  }

  if (action_mapping_) {
    // All actions are regarded as legal
    movelist.push_back(ActionType::kFold);
    movelist.push_back(ActionType::kCall);
    movelist.push_back(ActionType::kRaise);
    return movelist;
  }

  // Can't just randomly fold; only allow fold when under pressure.
  if (stakes_ > ante_[cur_player_]) {
    movelist.push_back(ActionType::kFold);
  }

  // Can always call/check
  movelist.push_back(ActionType::kCall);

  if (num_raises_ < kMaxRaises) {//raise次数(times of raise)
    movelist.push_back(ActionType::kRaise);
  }

  return movelist;
}

std::string LeducState::ActionToString(Player player, Action move) const {
  return GetGame()->ActionToString(player, move);
}

std::string LeducState::ToString() const {
  std::string result;

  absl::StrAppend(&result, "Round: ", round_, "\nPlayer: ", cur_player_,
                  "\nPot: ", pot_, "\nMoney (p1 p2 ...):");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", money_[p]);
  }
  absl::StrAppend(&result, "\nCards (public p1 p2 ...): ", public_card_, " ");
  for (Player player_index = 0; player_index < num_players_; player_index++) {
    absl::StrAppend(&result, private_cards_[player_index], " ");
  }

  absl::StrAppend(&result, "\nRound 1 sequence: ");
  for (int i = 0; i < round1_sequence_.size(); ++i) {
    Action action = round1_sequence_[i];
    if (i > 0) absl::StrAppend(&result, ", ");
    absl::StrAppend(&result, StatelessActionToString(action));
  }
  absl::StrAppend(&result, "\nRound 2 sequence: ");
  for (int i = 0; i < round2_sequence_.size(); ++i) {
    Action action = round2_sequence_[i];
    if (i > 0) absl::StrAppend(&result, ", ");
    absl::StrAppend(&result, StatelessActionToString(action));
  }
  absl::StrAppend(&result, "\n");

  return result;
}

bool LeducState::IsTerminal() const {
  
  //修改游戏结束判断条件
  return remaining_players_ == 1 || folded_[0] == true || (round_ == 2 && ReadyForNextRound());
}

std::vector<double> LeducState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  std::vector<double> returns(num_players_);
  for (auto player = Player{0}; player < num_players_; ++player) {
    // Money vs money at start.
    returns[player] = money_[player] - kStartingMoney;
  }

  return returns;
}

// Information state is card then bets.
std::string LeducState::InformationStateString(Player player) const {
  const LeducGame& game = open_spiel::down_cast<const LeducGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

// Observation is card then contribution of each players to the pot.
std::string LeducState::ObservationString(Player player) const {
  const LeducGame& game = open_spiel::down_cast<const LeducGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void LeducState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const LeducGame& game = open_spiel::down_cast<const LeducGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void LeducState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const LeducGame& game = open_spiel::down_cast<const LeducGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> LeducState::Clone() const {
  return std::unique_ptr<State>(new LeducState(*this));
}

std::vector<std::pair<Action, double>> LeducState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;

  if (suit_isomorphism_) {
    const double p = 1.0 / deck_size_;
    // Consecutive cards in deck are viewed identically.
    for (int card = 0; card < deck_.size() / 2; card++) {
      if (deck_[card * 2] != kInvalidCard &&
          deck_[card * 2 + 1] != kInvalidCard) {
        outcomes.push_back({card, p * 2});
      } else if (deck_[card * 2] != kInvalidCard ||
                 deck_[card * 2 + 1] != kInvalidCard) {
        outcomes.push_back({card, p});
      }
    }
    return outcomes;
  }

  const double p = 1.0 / deck_size_;
  for (int card = 0; card < deck_.size(); card++) {
    // This card is still in the deck, prob is 1/decksize.
    if (deck_[card] != kInvalidCard) outcomes.push_back({card, p});
  }
  return outcomes;
}

int LeducState::NextPlayer() const {
  // If we are on a chance node, it is the first player to play
  int current_real_player;
  if (cur_player_ == kChancePlayerId) {
    current_real_player = -1;
  } else {
    current_real_player = cur_player_;
  }
  // Go to the next player who's still in.
  for (int i = 1; i < num_players_; ++i) {
    Player player = (current_real_player + i) % num_players_;

    SPIEL_CHECK_TRUE(player >= 0);
    SPIEL_CHECK_TRUE(player < num_players_);
    if (!folded_[player]) {
      return player;
    }
  }

  SpielFatalError("Error in LeducState::NextPlayer(), should not get here.");
}

int LeducState::RankHand(Player player) const {
  int hand[] = {public_card_, private_cards_[player]};
  // Put the lower card in slot 0, the higher in slot 1.
  if (hand[0] > hand[1]) {
    std::swap(hand[0], hand[1]);
  }

  if (suit_isomorphism_) {
    int num_cards = deck_.size() / kNumSuits;
    if (hand[0] == hand[1]) {
      // Pair! Offset by deck_size_^2 to put higher than every singles combo.
      return (num_cards * num_cards + hand[0]);
    } else {
      // Otherwise card value dominates. Suit isomorphism has already removed
      // the distinction between suits, so we can compare the ranks directly.
      // This could lead to ties/draws and/or multiple winners.
      return hand[1] * num_cards + hand[0];
    }
  }

  // E.g. rank for two players:
  // 0 J1, 1 J2, 2 Q1, 3 Q2, 4 K1, 5 K2.
  int num_cards = deck_.size();

  if (hand[0] / kNumSuits == hand[1] / kNumSuits){
    // Pair! Offset by deck_size_^2 to put higher than every singles combo.
    return (num_cards * num_cards + hand[0] / kNumSuits);
  } else {
    // Otherwise card value dominates. No high/low suit: only two suits, and
    // given ordering above, dividing by gets the value (integer division
    // intended.) This could lead to ties/draws and/or multiple winners.
    return (hand[1] / kNumSuits) * num_cards + (hand[0] / kNumSuits);
  }
}

void LeducState::ResolveWinner() {
  num_winners_ = kInvalidPlayer;
  // std::cout<<"remaining_players_ "<<remaining_players_<<std::endl;
  if (remaining_players_ == 1) {
    // Only one left in? They get the pot!
    //游戏结束时候只剩余一人，如果这个人是对手那么对手胜并获得全部收益。
    //如果这个人是团队玩家则团队获胜并均分收益。
    for (Player player_index = 0; player_index < num_players_; player_index++) {
      if (!folded_[player_index]) {
        num_winners_ = 1;
        winner_[player_index] = true;
        if(player_index == 0){
          double sum = 0;
          money_[player_index] += pot_;
          for (Player player_index = 1; player_index < num_players_; player_index++){
            sum += money_[player_index];
          }
          for (Player player_index = 1; player_index < num_players_; player_index++){
            money_[player_index] = sum /(num_players_-1);
          }
        }else{
          double sum = 0;
          for (Player player_index = 1; player_index < num_players_; player_index++){
            money_[player_index] += static_cast<double>(pot_) /(num_players_-1);
            sum += money_[player_index];
          }
          for (Player player_index = 1; player_index < num_players_; player_index++){
            money_[player_index] = sum /(num_players_-1);
          }
        }
        pot_ = 0;
        return;
      }
    }

  } else {
    //其余情况为游戏结束时仍然剩余多名玩家，我们首先检查公共牌是否被翻开。

    //若公共牌没有被翻开，则对手玩家（玩家0 ）必然在round1选择了fold，那么团队获胜，团队内玩家均分收益。

    //若公共牌被翻开，则对手玩家（玩家0 ）必然没有在round1选择fold，那么最后游戏结束时剩余的是对手玩家和其他团队玩家。
    //我们通过对比手牌大小便可以知道谁是获胜者，若有多个胜者则算平局，返还全部筹码；若对手玩家获胜则他获得全部收益；若团队玩家获胜，团队内玩家均分收益。

    //新增判断检查公共牌是否翻开
    if(public_card_ == kInvalidCard){
      num_winners_ = 0;
      std::fill(winner_.begin(), winner_.end(), false);
      for (Player player_index = 1; player_index < num_players_; player_index++){
        if (!folded_[player_index]){
          winner_[player_index] = true;
          num_winners_++;
        }
      }
      double sum = 0;
      for (Player player_index = 1; player_index < num_players_; player_index++){
        money_[player_index] += static_cast<double>(pot_) /(num_players_-1);
        sum += money_[player_index];
      }
      for (Player player_index = 1; player_index < num_players_; player_index++){
        money_[player_index] = sum /(num_players_-1);
      }
      return;
    }

    int best_hand_rank = -1;
    num_winners_ = 0;
    std::fill(winner_.begin(), winner_.end(), false);
    for (Player player_index = 0; player_index < num_players_; player_index++) {
      if (!folded_[player_index]) {
        int rank = RankHand(player_index);
        if (rank > best_hand_rank) {
          // Beat the current best hand! Clear the winners list, then add.
          best_hand_rank = rank;
          std::fill(winner_.begin(), winner_.end(), false);
          winner_[player_index] = true;
          num_winners_ = 1;
        } else if (rank == best_hand_rank) {
          // Tied with best hand rank, so this player is a winner as well.
          winner_[player_index] = true;
          num_winners_++;
        }
      }
    }

    // Split the pot among the winners (possibly only one).
    SPIEL_CHECK_TRUE(1 <= num_winners_ && num_winners_ <= num_players_);
    
    //收益修改
    Player player_index = 0;
    if (winner_[player_index]){
        if(num_winners_ == 1){
            money_[player_index] += static_cast<double>(pot_);
            double sum = 0;
            for (Player player_index = 1; player_index < num_players_; player_index++){
              sum += money_[player_index];
            }
            for (Player player_index = 1; player_index < num_players_; player_index++){
              money_[player_index] = sum /(num_players_-1);
            }
        }else{
          for (Player player_index = 0; player_index < num_players_; player_index++){
            money_[player_index] = kStartingMoney;
          } 
        }
    }else{
      double sum = 0;
      for (Player player_index = 1; player_index < num_players_; player_index++){
        money_[player_index] += static_cast<double>(pot_) /(num_players_-1);
        sum += money_[player_index];
      }
      for (Player player_index = 1; player_index < num_players_; player_index++){
        money_[player_index] = sum /(num_players_-1);
      }
    }
    // for (Player player_index = 0; player_index < num_players_; player_index++) {
    //   if (winner_[player_index]) {
    //     // Give this player their share.
    //     money_[player_index] += static_cast<double>(pot_) / num_winners_;
    //   }
    // }
    pot_ = 0;
  }
}

bool LeducState::ReadyForNextRound() const {
  return ((num_raises_ == 0 && num_calls_ == remaining_players_) ||
          (num_raises_ > 0 && num_calls_ == (remaining_players_ - 1)));
}

void LeducState::NewRound() {
  SPIEL_CHECK_EQ(round_, 1);
  round_++;
  num_raises_ = 0;
  num_calls_ = 0;
  cur_player_ = kChancePlayerId;  // Public card.
}

void LeducState::SequenceAppendMove(int move) {
  if (round_ == 1) {
    round1_sequence_.push_back(move);
  } else {
    SPIEL_CHECK_EQ(round_, 2);
    round2_sequence_.push_back(move);
  }
}

void LeducState::Ante(Player player, int amount) {
  pot_ += amount;
  ante_[player] += amount;
  money_[player] -= amount;
}

std::vector<int> LeducState::padded_betting_sequence() const {
  std::vector<int> history = round1_sequence_;

  // We pad the history to the end of the first round with kPaddingAction.
  history.resize(game_->MaxGameLength() / 2, kInvalidAction);

  // We insert the actions that happened in the second round, and fill to
  // MaxGameLength.
  history.insert(history.end(), round2_sequence_.begin(),
                 round2_sequence_.end());
  history.resize(game_->MaxGameLength(), kInvalidAction);
  return history;
}

void LeducState::SetPrivate(Player player, Action move) {
  // Round 1. `move` refers to the card value to deal to the current
  // underlying player (given by `private_cards_dealt_`).
  if (suit_isomorphism_) {
    // Consecutive cards are identical under suit isomorphism.
    private_cards_[player] = move;
    for(int i=0; i<kNumSuits; i++){
      if (deck_[move * 2+i] != kInvalidCard){
        deck_[move * 2+i] = kInvalidCard;
        break;
      }
      if(i == kNumSuits-1){
        SpielFatalError("Suit isomorphism error.");
      }
    }
    // if (deck_[move * 2] != kInvalidCard) {
    //   deck_[move * 2] = kInvalidCard;
    // } else if (deck_[move * 2 + 1] != kInvalidCard) {
    //   deck_[move * 2 + 1] = kInvalidCard;
    // } else {
    //   SpielFatalError("Suit isomorphism error.");
    // }
  } else {
    private_cards_[player] = deck_[move];
    deck_[move] = kInvalidCard;
  }
  --deck_size_;
  ++private_cards_dealt_;

  // When all private cards are dealt, move to player 0.
  if (private_cards_dealt_ == num_players_) cur_player_ = 0;
}

std::unique_ptr<State> LeducState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> clone = game_->NewInitialState();

  // First, deal out cards:
  Action player_chance = history_.at(player_id).action;
  for (int p = 0; p < GetGame()->NumPlayers(); ++p) {
    if (p == player_id) {
      clone->ApplyAction(history_.at(p).action);
    } else {
      Action chosen_action = player_chance;
      while (chosen_action == player_chance || chosen_action == public_card_) {
        chosen_action = SampleAction(clone->ChanceOutcomes(), rng()).first;
      }
      clone->ApplyAction(chosen_action);
    }
  }
  for (int action : round1_sequence_) clone->ApplyAction(action);
  if (public_card_ != kInvalidCard) {
    clone->ApplyAction(public_card_);
    for (int action : round2_sequence_) clone->ApplyAction(action);
  }
  return clone;
}

int LeducState::NumObservableCards() const {
  return suit_isomorphism_ ? deck_.size() / kNumSuits : deck_.size();
}

//return (1 + kMaxRaises)*num_players_ - kMaxRaises;
int LeducState::MaxBetsPerRound() const { return (1+kMaxRaises) * num_players_ - kMaxRaises; }

void LeducState::SetPrivateCards(const std::vector<int>& new_private_cards) {
  SPIEL_CHECK_EQ(new_private_cards.size(), NumPlayers());
  private_cards_ = new_private_cards;
}

LeducGame::LeducGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      total_cards_((num_players_ + kNumRanks) * kNumSuits), //修改rank (KNumRanks>=0)
      action_mapping_(ParameterValue<bool>("action_mapping")),
      suit_isomorphism_(ParameterValue<bool>("suit_isomorphism")) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
  default_observer_ = std::make_shared<LeducObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<LeducObserver>(kInfoStateObsType);
}

std::unique_ptr<State> LeducGame::NewInitialState() const {
  return absl::make_unique<LeducState>(shared_from_this(),
                                       /*action_mapping=*/action_mapping_,
                                       /*suit_isomorphism=*/suit_isomorphism_);
}

int LeducGame::MaxChanceOutcomes() const {
  if (suit_isomorphism_) {
    return total_cards_ / kNumSuits;
  } else {
    return total_cards_;
  }
}

std::vector<int> LeducGame::InformationStateTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by maximum game length * 2 bits each (call / raise)
  if (suit_isomorphism_) {
    return {(num_players_) + (total_cards_) + (MaxGameLength() * 2)};
  } else {
    //return {(num_players_) + (total_cards_ * 2) + (MaxGameLength() * 2)};
    return {(num_players_) + (total_cards_ * num_players_ + total_cards_) + (MaxGameLength() * 2)};
  }
}

std::vector<int> LeducGame::ObservationTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by the contribution of each player to the pot
  if (suit_isomorphism_) {
    return {(num_players_) + (total_cards_) + (num_players_)};
  } else {
    //return {(num_players_) + (total_cards_ * 2) + (num_players_)};
    return {(num_players_) + (total_cards_ * num_players_ + total_cards_) + (num_players_)};
  }
}

double LeducGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus the money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.
  return (num_players_ - 1) * (kTotalRaisesPerRound * kFirstRaiseAmount +
                               kTotalRaisesPerRound * kSecondRaiseAmount + 1);
}

double LeducGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus the money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip
  // they put in to play.
  return -1 * (kTotalRaisesPerRound * kFirstRaiseAmount +
               kTotalRaisesPerRound * kSecondRaiseAmount + 1);
}

std::shared_ptr<Observer> LeducGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (params.empty()) {
    return std::make_shared<LeducObserver>(
        iig_obs_type.value_or(kDefaultObsType));
  } else {
    return MakeRegisteredObserver(iig_obs_type, params);
  }
}

std::string LeducGame::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance outcome:", action);
  } else {
    return StatelessActionToString(action);
  }
}

TabularPolicy GetAlwaysFoldPolicy(const Game& game) {
  SPIEL_CHECK_TRUE(
      dynamic_cast<LeducGame*>(const_cast<Game*>(&game)) != nullptr);
  return GetPrefActionPolicy(game, {ActionType::kFold, ActionType::kCall});
}

TabularPolicy GetAlwaysCallPolicy(const Game& game) {
  SPIEL_CHECK_TRUE(
      dynamic_cast<LeducGame*>(const_cast<Game*>(&game)) != nullptr);
  return GetPrefActionPolicy(game, {ActionType::kCall});
}

TabularPolicy GetAlwaysRaisePolicy(const Game& game) {
  SPIEL_CHECK_TRUE(
      dynamic_cast<LeducGame*>(const_cast<Game*>(&game)) != nullptr);
  return GetPrefActionPolicy(game, {ActionType::kRaise, ActionType::kCall});
}

}  // namespace leduc_poker
}  // namespace open_spiel