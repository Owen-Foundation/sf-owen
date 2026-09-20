#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>

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

    TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];
    NNZInfo<L1> nnzInfo;
    int bucket = (pos.count<ALL_PIECES>() - 1) / 4;

    net->featureTransformer.transform(pos, accumulators, caches, transformedFeatures, bucket, nnzInfo);

    std::cout << "Transformed Features [0..15]: ";
    for (int i = 0; i < 16; ++i) std::cout << (int)transformedFeatures[i] << " ";
    std::cout << "\nTransformed Features [512..527]: ";
    for (int i = 512; i < 528; ++i) std::cout << (int)transformedFeatures[i] << " ";
    std::cout << "\n";

    return 0;
}
