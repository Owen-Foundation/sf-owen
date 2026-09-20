# Plan: SF knowledge into Owen

## Phase 1 — loader (correctness first)
Parse `.nnue` into tensors; implement the feature transformer
(HalfKAv2_hm + FullThreats + PP_3Wide, 8 buckets/stacks) and the
SqrClippedReLU head in Python (torch or numpy). **Gate:** bitwise match
Stockfish's eval on 10k+ test positions.

## Phase 2 — fine-tune
Continue training the loaded net on Owen self-play data
(`sdata` format, WDL + PV-aware loss). Small LR, frozen buckets first,
then full unfreeze. Output stays `.nnue`-compatible.

## Phase 3 — engine merge
Owen loads `.nnue` natively: implement the SF forward pass in C++
( scalar first, SIMD later), selectable via `NNUEFile`, side by side
with `.o2nn`. Bench + perft + game validation.

## Phase 4 — surpass
Gate fine-tuned nets vs `o2` champions on DefenceTest (SPRT). Promote on
green. Iterate: distill match knowledge back, roll the loop.

## Legal
Stockfish is GPL-3.0; Owen is GPL-3.0. All reuse stays within GPL terms:
no proprietary weights, own training data, credit upstream.
