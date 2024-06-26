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

#include "open_spiel/games/liars_dice_mp/liars_dice_mp.h"

#include <algorithm>
#include <array>
#include <utility>

#include "open_spiel/game_parameters.h"

namespace open_spiel {
namespace liars_dice_mp {

namespace {
// Default Parameters.
constexpr int kDefaultPlayers = 3;//人数
constexpr int kDefaultNumDice = 1;//每人骰子数
constexpr int kDefaultDiceSides = 3;  //骰子面数
constexpr const char* kDefaultBiddingRule = "reset-face";
constexpr int kInvalidOutcome = -1;
constexpr int kInvalidBid = -1;

// Only relevant for the imperfect recall version.
constexpr int kDefaultRecallLength = 4;

// Facts about the game
const GameType kGameType{
    /*short_name=*/"liars_dice_mp",
    /*long_name=*/"Liars Dice Mp",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,//显性随机
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kDefaultPlayers,
    /*min_num_players=*/kDefaultPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"players", GameParameter(kDefaultPlayers)},
     {"numdice", GameParameter(kDefaultNumDice)},
     {"dice_sides", GameParameter(kDefaultDiceSides)},
     {"bidding_rule", GameParameter(kDefaultBiddingRule)}}};


std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LiarsDiceGame(params, kGameType));
}

const BiddingRule ParseBiddingRule(const std::string& bidding_rule_str) {
  SPIEL_CHECK_TRUE(bidding_rule_str == "reset-face" ||
                   bidding_rule_str == "reset-quantity");
  if (bidding_rule_str == "reset-face") {
    return BiddingRule::kResetFace;
  } else {
    return BiddingRule::kResetQuantity;
  }
}

const LiarsDiceGame* UnwrapGame(const Game* game) {
  return down_cast<const LiarsDiceGame*>(game);
}
}  // namespace

REGISTER_SPIEL_GAME(kGameType, Factory);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

//初始化state对象
//game: 指向当前游戏实例的智能指针，包含了游戏的全局设置和规则。
//total_num_dice: 游戏中所有玩家的骰子总数。
//max_dice_per_player: 单个玩家拥有的最大骰子数。
//num_dice: 一个整数向量，每个元素表示一个玩家拥有的骰子数量。
LiarsDiceState::LiarsDiceState(std::shared_ptr<const Game> game,
                               int total_num_dice, int max_dice_per_player,
                               const std::vector<int>& num_dice)
    : State(game),
      dice_outcomes_(),              //初始化骰子结果向量
      bidseq_(),                     //初始化出价序列
      cur_player_(kChancePlayerId),  // 设置当前玩家为机会玩家（表示游戏开始时掷骰子的阶段）。
      cur_roller_(0),                // 设置当前掷骰子的玩家为第一位玩家。
      winner_(kInvalidPlayer),       //初始化获胜者和失败者为无效值
      loser_(kInvalidPlayer),
      current_bid_(kInvalidBid),      //初始化当前出价为无效值。
      total_num_dice_(total_num_dice),//设置游戏中的骰子总数。
      total_moves_(0),                //初始化总行动次数为0
      calling_player_(0),             //初始化发起质疑的玩家和最后出价的玩家编号为0。
      bidding_player_(0),
      max_dice_per_player_(max_dice_per_player),//设置单个玩家拥有的最大骰子数。
      num_dice_(num_dice),            //将每个玩家的骰子数量复制到成员变量中。
      num_dice_rolled_(game->NumPlayers(), 0),//初始化一个向量，存储每位玩家已掷出的骰子数，初始值为0。
      bidseq_str_()                   //初始化出价序列字符串为空。
  {                 
  for (int const& num_dices : num_dice_) {//为每位玩家创建一个表示其骰子结果的向量，并将所有结果初始化为kInvalidOutcome（表示无效结果）。
    std::vector<int> initial_outcomes(num_dices, kInvalidOutcome);
    dice_outcomes_.push_back(initial_outcomes);
  }
}

//将游戏中的动作（由action_id参数指定）转换成易于理解的字符串表示
std::string LiarsDiceState::ActionToString(Player player,
                                           Action action_id) const {
  if (player != kChancePlayerId) {
    if (action_id == total_num_dice_ * dice_sides()) {
      return "Liar";
    } else {
      const std::pair<int, int> bid = UnrankBid(action_id);
      return absl::StrCat(bid.first, "-", bid.second);
    }
  }
  return absl::StrCat("Roll ", action_id + 1);
}

//返回当前玩家
int LiarsDiceState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

void LiarsDiceState::ResolveWinner() {
  //解析当前出价
  const std::pair<int, int> bid = UnrankBid(current_bid_);
  //得到出价的数量（quantity）和面值（face）。
  int quantity = bid.first, face = bid.second;


  //统计匹配的骰子数量
  int matches = 0;
  //遍历每个玩家(p)的每个骰子(d)，检查骰子的面值是否与出价的面值相同，或者是否为万能面（dice_sides()，例如6面骰子的6面是万能面，可以代表任何值）。
  //如果骰子面值匹配或为万能面，matches增加。
  for (auto p = Player{0}; p < num_players_; p++) {
    for (int d = 0; d < num_dice_[p]; d++) {
      if (dice_outcomes_[p][d] == face ||
          dice_outcomes_[p][d] == dice_sides()) {
        matches++;
      }
    }
  }

  //判断输赢
  //如果匹配的骰子数量至少等于出价的数量（即出价是真实的），则出价的玩家（bidding_player_）赢，发起质疑的玩家（calling_player_）输。
  if (matches >= quantity) {
    winner_ = bidding_player_;
    loser_ = calling_player_;
  } else {
    winner_ = calling_player_;
    loser_ = bidding_player_;
  }
}

//从游戏中获取骰子的面数
const int LiarsDiceState::dice_sides() const {
  return UnwrapGame(game_.get())->dice_sides();
}

//
//
const BiddingRule LiarsDiceState::bidding_rule() const {
  return UnwrapGame(game_.get())->bidding_rule();
}

//执行动作
void LiarsDiceState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    // 为当前roller摇下一个骰子
    SPIEL_CHECK_GE(cur_roller_, 0);
    SPIEL_CHECK_LT(cur_roller_, num_players_);

    SPIEL_CHECK_LT(num_dice_rolled_[cur_roller_], num_dice_[cur_roller_]);
    int slot = num_dice_rolled_[cur_roller_];

    // Assign the roll.
    dice_outcomes_[cur_roller_][slot] = action + 1;
    num_dice_rolled_[cur_roller_]++;

    //检查是否轮到下一个roller
    if (num_dice_rolled_[cur_roller_] == num_dice_[cur_roller_]) {
      cur_roller_++;
      if (cur_roller_ >= num_players_) {
        // Time to start playing!
        cur_player_ = 0;
        // Sort all players' rolls
        for (auto p = Player{0}; p < num_players_; p++) {
          std::sort(dice_outcomes_[p].begin(), dice_outcomes_[p].end());
        }
      }
    }
  } else {
    // 检查合法行动
    // if (!bidseq_.empty() && action <= bidseq_.back()) {
    //   SpielFatalError(absl::StrCat("Illegal action. ", action,
    //                                " should be strictly higher than ",
    //                                bidseq_.back()));
    // }
    if (action == total_num_dice_ * dice_sides()) {//如果选择了‘Liar’，游戏结束，准备分输赢
      bidseq_.push_back(action);
      calling_player_ = cur_player_;
      ResolveWinner();
    } else {
      //否则下位玩家继续出价
      bidseq_.push_back(action);
      current_bid_ = action;
      bidding_player_ = cur_player_;
      cur_player_ = NextPlayerRoundRobin(cur_player_, num_players_);
    }

    total_moves_++;
  }
}

//返回合法动作
std::vector<Action> LiarsDiceState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  }

  std::vector<Action> actions;
  //如果上一个玩家是队友且队友出价为最高，则当前合法动作为队友动作
  if(cur_player_ > 1){//若当前玩家为2,3,4...这种
    if(current_bid_ == total_num_dice_ * dice_sides() - 1){
      actions.push_back(current_bid_);
    }else{
      for (int b = current_bid_ + 1; b < total_num_dice_ * dice_sides(); b++) {
        actions.push_back(b);
      }
      // if (total_moves_ > 0) {
      //   actions.push_back(total_num_dice_ * dice_sides());
      // }
    }
  }else{
    for (int b = current_bid_ + 1; b < total_num_dice_ * dice_sides(); b++) {
      actions.push_back(b);
    }
    if (total_moves_ > 0) {
      actions.push_back(total_num_dice_ * dice_sides());
    }
  }
  return actions;
}

std::vector<std::pair<Action, double>> LiarsDiceState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());

  std::vector<std::pair<Action, double>> outcomes;

  // A chance node is a single die roll.
  outcomes.reserve(dice_sides());
  for (int i = 0; i < dice_sides(); i++) {
    outcomes.emplace_back(i, 1.0 / dice_sides());
  }

  return outcomes;
}

std::string LiarsDiceState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string result = absl::StrJoin(dice_outcomes_[player], "");
  for (int b = 0; b < bidseq_.size(); b++) {
    if (bidseq_[b] == total_num_dice_ * dice_sides()) {
      absl::StrAppend(&result, " Liar");
    } else {
      const std::pair<int, int> bid = UnrankBid(bidseq_[b]);
      absl::StrAppend(&result, " ", bid.first, "-", bid.second);
    }
  }
  return result;
}

std::string LiarsDiceState::ToString() const {
  std::string result = "";

  for (auto p = Player{0}; p < num_players_; p++) {
    if (p != 0) absl::StrAppend(&result, " ");
    for (int d = 0; d < num_dice_[p]; d++) {
      absl::StrAppend(&result, dice_outcomes_[p][d]);
    }
  }

  if (IsChanceNode()) {
    return absl::StrCat(result, " - chance node, current roller is player ",
                        cur_roller_);
  }

  for (int b = 0; b < bidseq_.size(); b++) {
    if (bidseq_[b] == total_num_dice_ * dice_sides()) {
      absl::StrAppend(&result, " Liar");
    } else {
      const std::pair<int, int> bid = UnrankBid(bidseq_[b]);
      absl::StrAppend(&result, " ", bid.first, "-", bid.second);
    }
  }
  return result;
}

bool LiarsDiceState::IsTerminal() const { return winner_ != kInvalidPlayer; }

std::vector<double> LiarsDiceState::Returns() const {
  std::vector<double> returns(num_players_, 0.0);

  if (winner_ != kInvalidPlayer) {
    if(winner_ == 0){
      returns[winner_] = 1.0;
      for(int player = 1; player < num_players_; player++){
        returns[player] = -1.0/(num_players_-1);
      }
    }else{
      returns[0] = -1.0;
      for(int player = 1; player < num_players_; player++){
        returns[player] = 1.0/(num_players_-1);
      }
    }
    
  }

  // if (loser_ != kInvalidPlayer) {
  //   returns[loser_] = -1.0;
  // }

  return returns;
}

void LiarsDiceState::InformationStateTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // 编码玩家的num_players位
  // 编码每位玩家每个骰子具体信息的(max_dice_per_player_ * dice_sides_)位
  // 编码合法动作的(total_num_dice_ * dice_sides_)位
  // 编码表示是否叫骰的1位

  //初始化了一个偏移量offset为0，并将values中的所有元素都设置为0
  int offset = 0;
  std::fill(values.begin(), values.end(), 0.);
  SPIEL_CHECK_EQ(values.size(), num_players_ +
                                    (max_dice_per_player_ * dice_sides()) +
                                    (total_num_dice_ * dice_sides()) + 1);
  
  //表示玩家编号
  values[player] = 1;
  offset += num_players_;

  //表示玩家的骰子信息
  int my_num_dice = num_dice_[player];

  for (int d = 0; d < my_num_dice; d++) {
    int outcome = dice_outcomes_[player][d];
    if (outcome != kInvalidOutcome) {
      SPIEL_CHECK_GE(outcome, 1);
      SPIEL_CHECK_LE(outcome, dice_sides());
      values[offset + (outcome - 1)] = 1;
    }
    offset += dice_sides();
  }
  
  // Skip to bidding part. If current player has fewer dice than the other
  // players, all the remaining entries are 0 for those dice.
  offset = num_players_ + max_dice_per_player_ * dice_sides();

  for (int b = 0; b < bidseq_.size(); b++) {
    SPIEL_CHECK_GE(bidseq_[b], 0);
    SPIEL_CHECK_LE(bidseq_[b], total_num_dice_ * dice_sides());
    values[offset + bidseq_[b]] = 1;
  }
}

void LiarsDiceState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // One-hot encoding for player number.
  // One-hot encoding for each die (max_dice_per_player_ * sides).
  // One slot(bit) for the two last legal bid.
  // One slot(bit) for calling liar. (Necessary because observations and
  // information states need to be defined at terminals)
  int offset = 0;
  std::fill(values.begin(), values.end(), 0.);
  SPIEL_CHECK_EQ(values.size(), num_players_ +
                                    (max_dice_per_player_ * dice_sides()) +
                                    (total_num_dice_ * dice_sides()) + 1);
  values[player] = 1;
  offset += num_players_;

  int my_num_dice = num_dice_[player];

  for (int d = 0; d < my_num_dice; d++) {
    int outcome = dice_outcomes_[player][d];
    if (outcome != kInvalidOutcome) {
      SPIEL_CHECK_GE(outcome, 1);
      SPIEL_CHECK_LE(outcome, dice_sides());
      values[offset + (outcome - 1)] = 1;
    }
    offset += dice_sides();
  }

  // Skip to bidding part. If current player has fewer dice than the other
  // players, all the remaining entries are 0 for those dice.
  offset = num_players_ + max_dice_per_player_ * dice_sides();

  // We only show the num_players_ last bids
  int size_bid = bidseq_.size();
  int bid_offset = std::max(0, size_bid - num_players_);
  for (int b = bid_offset; b < size_bid; b++) {
    SPIEL_CHECK_GE(bidseq_[b], 0);
    SPIEL_CHECK_LE(bidseq_[b], total_num_dice_ * dice_sides());
    values[offset + bidseq_[b]] = 1;
  }
}

std::unique_ptr<State> LiarsDiceState::Clone() const {
  return std::unique_ptr<State>(new LiarsDiceState(*this));
}

//铜锅bidnum获取出价细节
std::pair<int, int> LiarsDiceState::UnrankBid(int bidnum) const {
  std::pair<int, int> bid;
  SPIEL_CHECK_NE(bidnum, kInvalidBid);
  SPIEL_CHECK_GE(bidnum, 0);
  SPIEL_CHECK_LT(bidnum, dice_sides() * total_num_dice_);

  if (bidding_rule() == BiddingRule::kResetFace) {
    // Bids have the form <quantity>-<face>
    //
    // So, in a two-player game where each die has 6 faces, we have
    //
    // Bid ID    Quantity   Face
    // 0         1          1
    // 1         1          2
    // ...
    // 5         1          6
    // 6         2          1
    // ...
    // 11        2          6
    //
    // Bid ID #dice * #num faces encodes the special "liar" action.

    // The quantity occupies the higher bits, so it can be extracted using an
    // integer division operation.
    bid.first = bidnum / dice_sides() + 1;
    // The face occupies the lower bits, so it can be extraced using a modulo
    // operation.
    bid.second = 1 + (bidnum % dice_sides());
  } else {
    SPIEL_CHECK_EQ(bidding_rule(), BiddingRule::kResetQuantity);
    // Bids have the form <face>-<quantity>
    //
    // So, in a two-player game where each die has 6 faces, we have
    //
    // Bid ID    Quantity   Face
    // 0         1          1
    // 1         2          1
    // 2         1          2
    // 3         2          2
    // ...
    // 9         2          5
    // 10        1          6
    // 11        2          6
    //
    // Bid ID #dice * #num faces encodes the special "liar" action.
    //
    // This particular encoding scheme allows for very cheap comparison of bids:
    // a bid is stronger if it is encoded to a higher ID.

    // The quantity occupies the lower bits, so it can be extracted using a
    // modulo operation.
    bid.first = 1 + (bidnum % total_num_dice_);
    // The face occupies the higher bits, so it can be extracted using an
    // integer division.
    bid.second = bidnum / total_num_dice_ + 1;
  }

  SPIEL_CHECK_GE(bid.first, 1);
  // It doesn't make sense to bid more dice than the number of dice in the game.
  SPIEL_CHECK_LE(bid.first, total_num_dice_);

  SPIEL_CHECK_GE(bid.second, 1);
  // It doesn't make sense to bid a face that does not exist.
  SPIEL_CHECK_LE(bid.second, dice_sides());

  return bid;
}

LiarsDiceGame::LiarsDiceGame(const GameParameters& params, GameType game_type)
    : Game(game_type, params),
      num_players_(ParameterValue<int>("players")),
      dice_sides_(ParameterValue<int>("dice_sides")),
      bidding_rule_(ParseBiddingRule(
          ParameterValue<std::string>("bidding_rule"))) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
  SPIEL_CHECK_GE(dice_sides_, 1);

  int def_num_dice = ParameterValue<int>("numdice");

  // Compute the number of dice for each player based on parameters,
  // and set default outcomes of unknown face values (-1).
  total_num_dice_ = 0;
  num_dice_.resize(num_players_, 0);

  for (auto p = Player{0}; p < num_players_; p++) {
    std::string key = absl::StrCat("numdice", p);

    int my_num_dice = def_num_dice;
    if (IsParameterSpecified(game_parameters_, key)) {
      my_num_dice = ParameterValue<int>(key);
    }

    num_dice_[p] = my_num_dice;
    total_num_dice_ += my_num_dice;
  }

  // Compute max dice per player (used for observations.)
  max_dice_per_player_ = -1;
  for (int nd : num_dice_) {
    if (nd > max_dice_per_player_) {
      max_dice_per_player_ = nd;
    }
  }
}

int LiarsDiceGame::NumDistinctActions() const {
  return total_num_dice_ * dice_sides_ + 1;
}

std::unique_ptr<State> LiarsDiceGame::NewInitialState() const {
  std::unique_ptr<LiarsDiceState> state(
      new LiarsDiceState(shared_from_this(),
                         /*total_num_dice=*/total_num_dice_,
                         /*max_dice_per_player=*/max_dice_per_player_,
                         /*num_dice=*/num_dice_));
  return state;
}

int LiarsDiceGame::MaxChanceOutcomes() const { return dice_sides_; }

int LiarsDiceGame::MaxGameLength() const {
  // A bet for each side and number of total dice, plus "liar" action.
  return total_num_dice_ * dice_sides_ + 1;
}
int LiarsDiceGame::MaxChanceNodesInHistory() const { return total_num_dice_; }

std::vector<int> LiarsDiceGame::InformationStateTensorShape() const {
  // 编码玩家的num_players位
  // 编码每位玩家每个骰子具体信息的(max_dice_per_player_ * dice_sides_)位
  // 编码合法动作的(total_num_dice_ * dice_sides_)位
  // 编码calling liar玩家的1位
  return {num_players_ + (max_dice_per_player_ * dice_sides_) +
          (total_num_dice_ * dice_sides_) + 1};
}

std::vector<int> LiarsDiceGame::ObservationTensorShape() const {
  // One-hot encoding for the player number.
  // One-hot encoding for each die (max_dice_per_player_ * sides).
  // One slot(bit) for the num_players_ last legal bid.
  // One slot(bit) for calling liar. (Necessary because observations and
  // information states need to be defined at terminals)
  return {num_players_ + (max_dice_per_player_ * dice_sides_) +
          (total_num_dice_ * dice_sides_) + 1};
}

}  // namespace liars_dice_mp
}  // namespace open_spiel
