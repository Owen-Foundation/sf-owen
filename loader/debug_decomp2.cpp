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
#include "/home/hemesh/sf-src/src/nnue/features/half_ka_v2_hm.h"
#include "/home/hemesh/sf-src/src/nnue/features/full_threats.h"
#include "/home/hemesh/sf-src/src/nnue/features/pp_3wide.h"
#undef private

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;
using namespace Stockfish::Eval::NNUE::Features;

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

    Color us = pos.side_to_move();
    Square ksq_us = pos.square<KING>(us);

    int halfka_sum12 = 0, halfka_sum524 = 0;
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        Piece pc = pos.piece_on(s);
        if (pc != NO_PIECE && type_of(pc) != KING) {
            auto idx = HalfKAv2_hm::make_index(us, s, pc, ksq_us);
            halfka_sum12 += net->featureTransformer.weights[idx * 1024 + 12];
            halfka_sum524 += net->featureTransformer.weights[idx * 1024 + 524];
        }
    }
    std::cout << "C++ HalfKA sum j=12: " << halfka_sum12 << " j=524: " << halfka_sum524 << "\n";

    FullThreats::IndexList threats;
    FullThreats::append_active_indices(us, pos, threats);
    int threat_sum12 = 0, threat_sum524 = 0;
    for (auto idx : threats) {
        threat_sum12 += net->featureTransformer.threatWeightData()[idx * 1024 + 12];
        threat_sum524 += net->featureTransformer.threatWeightData()[idx * 1024 + 524];
    }
    std::cout << "C++ Threat sum j=12: " << threat_sum12 << " j=524: " << threat_sum524 << "\n";

    PP_3Wide::IndexList pps;
    PP_3Wide::append_active_indices(us, pos, pps);
    int pp_sum12 = 0, pp_sum524 = 0;
    for (auto idx : pps) {
        pp_sum12 += net->featureTransformer.pawnPairWeightData()[(idx - 59808) * 1024 + 12];
        pp_sum524 += net->featureTransformer.pawnPairWeightData()[(idx - 59808) * 1024 + 524];
    }
    std::cout << "C++ PP sum j=12: " << pp_sum12 << " j=524: " << pp_sum524 << "\n";

    return 0;
}
