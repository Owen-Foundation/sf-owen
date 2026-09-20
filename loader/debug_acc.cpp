#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#define private public
#include "/home/hemesh/sf-src/src/types.h"
#include "/home/hemesh/sf-src/src/position.h"
#include "/home/hemesh/sf-src/src/attacks.h"
#include "/home/hemesh/sf-src/src/nnue/network.h"
#include "/home/hemesh/sf-src/src/nnue/nnue_accumulator.h"
#undef private

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;

int main() {
    Attacks::init();
    Position::init();
    auto net = std::make_unique<Network>();
    EvalFile evalFile;
    evalFile.current = std::nullopt;
    net->load("", "/home/hemesh/sf-nets/nn-134a887f4c8f.nnue", evalFile);

    StateInfo si;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &si);

    AccumulatorStack accumulators;
    AccumulatorCaches caches(*net);
    accumulators.evaluate(pos, net->featureTransformer, caches);
    const auto& accState = accumulators.latest();

    Color us = pos.side_to_move();
    std::cout << "C++ Accumulator us [12..14]:\n";
    for (int j = 12; j <= 14; ++j) {
        int s0 = accState.accumulation[us][j];
        int s1 = accState.accumulation[us][j + 512];
        std::cout << "  j=" << j << ": sum0=" << s0 << " sum1=" << s1 << " prod=" << (s0*s1)/512 << "\n";
    }

    return 0;
}
