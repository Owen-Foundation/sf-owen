#include <iostream>
#include <memory>
#include "/home/hemesh/sf-src/src/types.h"
#include "/home/hemesh/sf-src/src/position.h"
#include "/home/hemesh/sf-src/src/bitboard.h"
#include "/home/hemesh/sf-src/src/attacks.h"
#include "/home/hemesh/sf-src/src/evaluate.h"
#include "/home/hemesh/sf-src/src/nnue/network.h"
#include "/home/hemesh/sf-src/src/nnue/nnue_accumulator.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;

int main() {
    Attacks::init();
    Position::init();
    auto net = std::make_unique<Network>();
    EvalFile evalFile;
    evalFile.current = std::nullopt;
    net->load("", "/home/hemesh/sf-nets/nn-134a887f4c8f.nnue", evalFile);

    if (!evalFile.current.has_value()) {
        std::cerr << "Failed to load net\n";
        return 1;
    }

    StateInfo si;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &si);

    AccumulatorStack accumulators;
    AccumulatorCaches caches(*net);

    auto trace = net->trace_evaluate(pos, accumulators, caches);
    std::cout << "Correct bucket: " << trace.correctBucket << "\n";
    for (int b = 0; b < 8; ++b) {
        std::cout << "Bucket " << b << ": psqt=" << trace.psqt[b] << " pos=" << trace.positional[b] << "\n";
    }
    return 0;
}
