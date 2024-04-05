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

#ifndef OPEN_SPIEL_GAMES_LIARS_DICE_INFO_H_
#define OPEN_SPIEL_GAMES_LIARS_DICE_INFO_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// A simple game that includes chance and imperfect information
// https://en.wikipedia.org/wiki/Liar%27s_dice
//
// Currently only supports a single round and two players.
// 当前规则中，最高面是万能的
//
// 参数:
//   "bidding_rule" string   出价规则（"reset-face"或"reset-quantity"）(默认 "reset-face")
//   "dice_sides"   int      每个骰子的面数            (默认 = 6)
//   "numdice"      int      每位玩家的骰子数              (默认 = 1)
//   "numdiceX"     int      特定玩家X的骰子数覆盖值 (默认 = 1)
//   "players"      int      玩家数                      (默认 = 2)

namespace open_spiel {
namespace liars_dice_info {

enum BiddingRule {
  // The player may bid a higher quantity of any particular face, or the same
  // quantity of a higher face (allowing a player to "re-assert" a face value
  // they believe prevalent if another player increased the face value on their
  // bid).
  kResetFace = 1,

  // The player may bid a higher quantity of the same face, or any particular
  // quantity of a higher face (allowing a player to "reset" the quantity).
  kResetQuantity = 2
};

class LiarsDiceGame;

class LiarsDiceState : public State {
 public:
  explicit LiarsDiceState(std::shared_ptr<const Game> game, int total_num_dice,
                          int max_dice_per_player,
                          const std::vector<int>& num_dice);
  LiarsDiceState(const LiarsDiceState&) = default;

  void Reset(const GameParameters& params);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;

  // 明确地从游戏中获取骰子的面数
  const int dice_sides() const;

  // 返回发起质疑的玩家。
  Player calling_player() const { return calling_player_; }

  // 返回给定玩家和骰子索引的骰子结果。
  int dice_outcome(Player player, int index) const {
    return dice_outcomes_[player][index];
  }

  // 返回最后的出价。如果最后一个出价是总骰子数*骰子面数，那么返回倒数第二个出价。
  int last_bid() const {
    if (bidseq_.back() == total_num_dice_ * dice_sides()) {
      return bidseq_[bidseq_.size() - 2];
    } else {
      return bidseq_.back();
    }
  }

 protected:
  void DoApplyAction(Action action_id) override;

  // 从整数中获取出价的数量和面值。返回的格式依赖于出价规则。
  // 出价从0开始，到total_dice*dice_sides-1（包含）。
  std::pair<int, int> UnrankBid(int bid) const;

  // 骰子结果：首先按玩家索引，然后是骰子编号。
  std::vector<std::vector<int>> dice_outcomes_;

  // 出价序列
  std::vector<int> bidseq_;

 private:
  void ResolveWinner();

  // 返回游戏使用的出价规则。
  const BiddingRule bidding_rule() const;

  // 初始化为无效值。使用Game::NewInitialState()。
  Player cur_player_;  // 轮到谁出牌。
  int cur_roller_;     // 当前正在掷骰子的玩家。
  int winner_;
  int loser_;
  int current_bid_;
  int total_num_dice_;
  int total_moves_;
  int calling_player_;  // 声明“说谎”的玩家。
  int bidding_player_;  // 最后出价的玩家。
  int max_dice_per_player_;

  std::vector<int> num_dice_;         // 每位玩家拥有的骰子数。
  std::vector<int> num_dice_rolled_;  // 当前已掷出的骰子数。

  // 用于编码信息状态。
  std::string bidseq_str_;
};

class LiarsDiceGame : public Game {
 public:
  explicit LiarsDiceGame(const GameParameters& params, GameType game_type);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;
  int MaxChanceNodesInHistory() const override;

  // 返回每位玩家拥有的最大骰子数。例如，如果玩家1有3个骰子，玩家2有2个
  int max_dice_per_player() const { return max_dice_per_player_; }

  // 返回骰子总数
  int total_num_dice() const { return total_num_dice_; }

  // 返回每人拥有的骰子数
  std::vector<int> num_dice() const { return num_dice_; }

  const int dice_sides() const { return dice_sides_; }
  const BiddingRule bidding_rule() const { return bidding_rule_; }

 private:
  // 玩家人数
  int num_players_;

  // 游戏中的总骰子数，决定合法出价。
  int total_num_dice_;

  std::vector<int> num_dice_;  // 每个玩家有多少骰子。
  int max_dice_per_player_;    // num_dice_ vector中的最大值。
  const int dice_sides_;       // 骰子面数
  const BiddingRule bidding_rule_;
};


}  // namespace liars_dice_info
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LIARS_DICE_MP_H_
