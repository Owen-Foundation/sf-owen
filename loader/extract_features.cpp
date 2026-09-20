#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <filesystem>
#include <array>

#define private public
#include "types.h"
#include "position.h"
#include "bitboard.h"
#include "attacks.h"
#include "nnue/features/half_ka_v2_hm.h"
#include "nnue/features/full_threats.h"
#include "nnue/features/pp_3wide.h"
#undef private

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

static const Stockfish::Piece OWEN_TO_SF_PIECE[13] = {
    Stockfish::W_PAWN, Stockfish::W_KNIGHT, Stockfish::W_BISHOP, Stockfish::W_ROOK, Stockfish::W_QUEEN, Stockfish::W_KING,
    Stockfish::B_PAWN, Stockfish::B_KNIGHT, Stockfish::B_BISHOP, Stockfish::B_ROOK, Stockfish::B_QUEEN, Stockfish::B_KING,
    Stockfish::NO_PIECE
};

void init_sf_bitboards() {
    Attacks::init();
    Position::init();
}

static inline void populate_features(const Position& pos, BoardFeatures* out) {
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
}

int extract_features_fen(const char* fen, BoardFeatures* out) {
    StateInfo si;
    Position pos;
    pos.set(fen, false, &si);
    populate_features(pos, out);
    return 0;
}

int extract_features_raw_board(const uint8_t* raw_board64, int stm, BoardFeatures* out) {
    StateInfo si;
    std::memset(&si, 0, sizeof(StateInfo));
    Position pos;
    pos.st = &si;
    pos.gamePly = 0;
    pos.sideToMove = Stockfish::Color(stm);

    std::memset(pos.pieceCount, 0, sizeof(pos.pieceCount));
    pos.byTypeBB.fill(0);
    pos.byColorBB.fill(0);
    pos.board.fill(Stockfish::NO_PIECE);

    for (int s = 0; s < 64; ++s) {
        uint8_t opc = raw_board64[s];
        if (opc < 12) {
            Stockfish::Piece spc = OWEN_TO_SF_PIECE[opc];
            Stockfish::Square ssq = Stockfish::Square(s);
            Stockfish::Color sc = Stockfish::color_of(spc);
            Stockfish::PieceType spt = Stockfish::type_of(spc);

            pos.board[ssq] = spc;
            pos.byTypeBB[Stockfish::ALL_PIECES] |= Stockfish::square_bb(ssq);
            pos.byTypeBB[spt] |= Stockfish::square_bb(ssq);
            pos.byColorBB[sc] |= Stockfish::square_bb(ssq);
            pos.pieceCount[spc]++;
        }
    }
    pos.set_state();
    populate_features(pos, out);
    return 0;
}

}
