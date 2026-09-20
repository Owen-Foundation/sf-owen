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

    std::cout << "C++ Bias j=12: " << (int)net->featureTransformer.biases[12] << " j=524: " << (int)net->featureTransformer.biases[524] << "\n";
    return 0;
}
