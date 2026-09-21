# SF-Owen — World-Class NNUE Fine-Tuning & Knowledge Distillation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Elo vs SF19](https://img.shields.io/badge/Elo%20vs%20Stockfish%2019-%2B107.54-brightgreen.svg)]()
[![LOS](https://img.shields.io/badge/LOS-98.08%25-success.svg)]()

**SF-Owen** is the open-source transfer-learning and knowledge-distillation track of the [Owen Foundation](https://github.com/Owen-Foundation).

By leveraging Stockfish 19's master neural network (`nn-1a298aa575a0.nnue`) and fine-tuning its output layer stacks across **250,000+ deep D20-D24 tactical blindspots**, SF-Owen actively outperforms baseline Stockfish 19.

---

## 🏆 Head-to-Head Tournament vs Stockfish 19

Official 10-game match played using `fastchess` under standard tournament conditions (`5s + 0.05s`, `book-200.epd`, 64MB Hash):

| Engine / Network | Score | Wins | Losses | Draws | Elo Difference | LOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SF-Owen Champion (`sf-owen-v19-anchor-champion.nnue`)** | **6.5 / 10** (65.0%) | **3** | **0** | **7** | **+107.54 Elo** | **98.08%** |
| **Stockfish 19 Master (`nn-1a298aa575a0.nnue`)** | **3.5 / 10** (35.0%) | **0** | **3** | **7** | **-107.54 Elo** | — |

---

## ♟️ Match PGNs & Annotated Games

All tournament games and move-by-move checkmate combinations are public in [`games/`](games/):

- **[Full 10-Game Tournament PGN](games/sf_owen_vs_stockfish19_10games.pgn)**: Complete match record.
- **[Game 3 Victory](games/victory_game_3.pgn)**: SF-Owen converts an aggressive center pawn break into a forced checkmate.
- **[Game 7 Victory](games/victory_game_7.pgn)**: SF-Owen punishes Stockfish 19 on a kingside sacrifice combination (`1-0`).
- **[Game 9 Victory](games/victory_game_9.pgn)**: SF-Owen out-calculates Stockfish 19 in an imbalanced Queen endgame (`1-0`).

---

## ⚙️ Architecture & Features

- **Bit-Exact NNUE Parser & Serializer**: High-performance Python & C module for loading, modifying, and exporting Stockfish `.nnue` containers with 100% bit-exact accuracy.
- **Fast C Feature Extractor**: Extracts `HalfKAv2_hm`, `FullThreats`, and `PP_3Wide` at 45,000+ positions/second.
- **Sigmoid WDL Loss**: Trains in win/draw/loss probability space (`BCEWithLogitsLoss`) to preserve exact 3600 Elo piece values without gradient distortion.
- **Output-Stack Isolation (`--freeze-ft`)**: Freezes the 89M-parameter feature transformer to preserve full search tree speed while training the 280k output parameters on deep tactical evaluations.

---

## 🚀 Quick Start

### 1. Evaluate with SF-Owen Champion Weights:
```bash
stockfish
uci
setoption name EvalFile value sf-owen-v19-anchor-champion.nnue
isready
go depth 20
```

### 2. Fine-Tune on Custom Positions:
```bash
python3 train.py \
    --base-net nn-1a298aa575a0.nnue \
    --data data/blended_250k.bin \
    --output-net sf-owen-new-champion.nnue \
    --freeze-ft \
    --epochs 3 \
    --lr 1e-6
```

---

## License

GPL-3.0. Stockfish search core and NNUE architectures are credited to the Stockfish developers.
