#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include "/home/hemesh/sf-src/src/types.h"
#include "/home/hemesh/sf-src/src/position.h"
#include "/home/hemesh/sf-src/src/bitboard.h"
#include "/home/hemesh/sf-src/src/attacks.h"
#include "/home/hemesh/sf-src/src/nnue/features/half_ka_v2_hm.h"
#include "/home/hemesh/sf-src/src/nnue/features/full_threats.h"
#include "/home/hemesh/sf-src/src/nnue/features/pp_3wide.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE::Features;

extern "C" {

struct FeatureList {
    int count;
    uint32_t indices[512];
};

struct BoardFeatures {
    int bucket;
    int side_to_move;
    FeatureList halfka_us;
    FeatureList halfka_them;
    FeatureList threat_us;
    FeatureList threat_them;
    FeatureList pp_us;
    FeatureList pp_them;
};

void init_sf_bitboards() {
    Attacks::init();
    Position::init();
}

int extract_features_fen(const char* fen, BoardFeatures* out) {
    StateInfo si;
    Position pos;
    pos.set(fen, false, &si);

    Color us = pos.side_to_move();
    Color them = ~us;
    out->side_to_move = (int)us;
    out->bucket = (pos.count<ALL_PIECES>() - 1) / 4;

    // 1. HalfKAv2_hm
    out->halfka_us.count = 0;
    out->halfka_them.count = 0;
    Square ksq_us = pos.square<KING>(us);
    Square ksq_them = pos.square<KING>(them);

    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        Piece pc = pos.piece_on(s);
        if (pc != NO_PIECE) {
            out->halfka_us.indices[out->halfka_us.count++] = HalfKAv2_hm::make_index(us, s, pc, ksq_us);
            out->halfka_them.indices[out->halfka_them.count++] = HalfKAv2_hm::make_index(them, s, pc, ksq_them);
        }
    }

    // 2. FullThreats
    FullThreats::IndexList threat_us_list, threat_them_list;
    FullThreats::append_active_indices(us, pos, threat_us_list);
    FullThreats::append_active_indices(them, pos, threat_them_list);

    out->threat_us.count = threat_us_list.size();
    for (size_t i = 0; i < threat_us_list.size(); ++i) {
        out->threat_us.indices[i] = threat_us_list[i];
    }

    out->threat_them.count = threat_them_list.size();
    for (size_t i = 0; i < threat_them_list.size(); ++i) {
        out->threat_them.indices[i] = threat_them_list[i];
    }

    // 3. PP_3Wide
    PP_3Wide::IndexList pp_us_list, pp_them_list;
    PP_3Wide::append_active_indices(us, pos, pp_us_list);
    PP_3Wide::append_active_indices(them, pos, pp_them_list);

    out->pp_us.count = pp_us_list.size();
    for (size_t i = 0; i < pp_us_list.size(); ++i) {
        out->pp_us.indices[i] = pp_us_list[i];
    }

    out->pp_them.count = pp_them_list.size();
    for (size_t i = 0; i < pp_them_list.size(); ++i) {
        out->pp_them.indices[i] = pp_them_list[i];
    }

    return 0;
}

}
