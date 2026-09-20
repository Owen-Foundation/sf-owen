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

    TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];
    NNZInfo<L1> nnzInfo;
    int bucket = (pos.count<ALL_PIECES>() - 1) / 4;

    net->featureTransformer.transform(pos, accumulators, caches, transformedFeatures, bucket, nnzInfo);

    const auto& arch = net->network[bucket];
    decltype(arch.fc_0)::OutputBuffer fc_0_out;
    decltype(arch.ac_sqr_0)::OutputType concat_buffer[32 * 4];
    decltype(arch.fc_1)::OutputBuffer fc_1_out;
    decltype(arch.fc_2)::OutputBuffer fc_2_out;

    arch.fc_0.propagate(transformedFeatures, fc_0_out, nnzInfo);
    arch.ac_sqr_0.propagate(fc_0_out, concat_buffer);
    arch.ac_0.propagate(fc_0_out, concat_buffer + 32);

    std::cout << "C++ fc0_out [0..7]: ";
    for (int i = 0; i < 8; ++i) std::cout << fc_0_out[i] << " ";
    std::cout << "\nC++ fc0_out [30, 31]: " << fc_0_out[30] << " " << fc_0_out[31] << "\n";

    std::cout << "C++ concat0 sqr [0..7]: ";
    for (int i = 0; i < 8; ++i) std::cout << (int)concat_buffer[i] << " ";
    std::cout << "\nC++ concat0 clip [0..7]: ";
    for (int i = 0; i < 8; ++i) std::cout << (int)concat_buffer[32 + i] << " ";
    std::cout << "\n";

    arch.fc_1.propagate(concat_buffer, fc_1_out);
    arch.ac_sqr_1.propagate(fc_1_out, concat_buffer + 64);
    arch.ac_1.propagate(fc_1_out, concat_buffer + 96);

    std::cout << "C++ fc1_out [0..7]: ";
    for (int i = 0; i < 8; ++i) std::cout << fc_1_out[i] << " ";
    std::cout << "\n";

    arch.fc_2.propagate(concat_buffer, fc_2_out);
    std::cout << "C++ fc2_out[0]: " << fc_2_out[0] << "\n";

    int fwdOut = fc_2_out[0] + (fc_0_out[30] - fc_0_out[31]);
    int positional = (fwdOut * 600 * 16) / (128 * 64 * 2);
    std::cout << "C++ fwdOut: " << fwdOut << " -> positional: " << positional << "\n";

    return 0;
}
