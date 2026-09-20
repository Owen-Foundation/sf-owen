#include <iostream>
#include <fstream>
#include <string>
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

extern "C" {

struct NativeEvalResult {
    int psqt;
    int positional;
    int combined_nnue;
    int blended_eval;
};

static Network* g_network = nullptr;

int init_native_network(const char* net_path) {
    Attacks::init();
    Position::init();
    if (!g_network) {
        g_network = new Network();
    }
    EvalFile evalFile;
    evalFile.current = std::nullopt;
    g_network->load("", net_path, evalFile);
    return evalFile.current.has_value() ? 0 : -1;
}

int evaluate_fen_native(const char* fen, NativeEvalResult* out) {
    if (!g_network) return -1;
    StateInfo si;
    Position pos;
    pos.set(fen, false, &si);

    AccumulatorStack accumulators;
    AccumulatorCaches caches(*g_network);

    auto [psqt, positional] = g_network->evaluate(pos, accumulators, caches);
    out->psqt = (int)psqt;
    out->positional = (int)positional;
    out->combined_nnue = (int)(psqt + positional);
    out->blended_eval = (int)Eval::evaluate(*g_network, pos, accumulators, caches, 0);
    return 0;
}

}
