#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game_name, "kuhn_poker_mp", "Game to run CFR on.");
ABSL_FLAG(int, num_iters, 1000, "How many iters to run for.");
ABSL_FLAG(int, report_every, 100, "How often to report exploitability.");

// Example code for using CFR+ to solve Kuhn Poker.
int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(absl::GetFlag(FLAGS_game_name));
  open_spiel::algorithms::CFRSolver solver(*game);
  std::cerr << "Starting CFR on " << game->GetType().short_name
            << "..." << std::endl;

  for (int i = 0; i < absl::GetFlag(FLAGS_num_iters); ++i) {
    solver.EvaluateAndUpdatePolicy();
    if (i % absl::GetFlag(FLAGS_report_every) == 0 ||
        i == absl::GetFlag(FLAGS_num_iters) - 1) {
      double exploitability = open_spiel::algorithms::Exploitability(
          *game, *solver.AveragePolicy());
      std::cerr << "Iteration " << i << " exploitability=" << exploitability
                << std::endl;
    }
  }
}
