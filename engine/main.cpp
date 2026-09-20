#include <iostream>
#include <memory>
#include <utility>

#include "/home/hemesh/sf-src/src/attacks.h"
#include "/home/hemesh/sf-src/src/misc.h"
#include "/home/hemesh/sf-src/src/position.h"
#include "/home/hemesh/sf-src/src/tune.h"
#include "/home/hemesh/sf-src/src/uci.h"

using namespace Stockfish;

int main(int argc, char* argv[]) {
    std::cout << "SF-Owen 1.0 by Owen Foundation" << std::endl;

    Attacks::init();
    Position::init();

    auto cli = CommandLine(argc, argv);
    auto uci = std::make_unique<UCIEngine>(std::move(cli));

    Tune::init(uci->engine_options());

    uci->loop();

    return 0;
}
