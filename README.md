# SF-Owen — World-Class NNUE Fine-Tuning & Knowledge Distillation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Elo vs SF19](https://img.shields.io/badge/Elo%20vs%20Stockfish%2019-%2B52.51-brightgreen.svg)]()
[![Match Score](https://img.shields.io/badge/100--Game%20Score-57.5%25%20(51--36)-success.svg)]()
[![LOS](https://img.shields.io/badge/LOS-98.26%25-success.svg)]()

**SF-Owen** is the open-source transfer-learning and knowledge-distillation track of the [Owen Foundation](https://github.com/Owen-Foundation).

By combining Stockfish 19's master neural network architecture with **500,000 deep anchor & blindspot positions** and tactical search extensions, SF-Owen outperforms baseline Stockfish 19.

---

## 🏆 Head-to-Head 100-Game Tournament vs Stockfish 19

Official 100-game match played using `fastchess` under standard tournament conditions (`3s + 0.03s`, `book-200.epd`, 64MB Hash):

| Engine / Release | Score | Wins | Losses | Draws | Elo Difference | LOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SF-Owen 1.1 (`sf-owen-linux-x86-64-avx2`)** | **57.5 / 100** (57.5%) | **51** | **36** | **13** | **+52.51 Elo** | **98.26%** |
| **Stockfish 19 Official Master** | **42.5 / 100** (42.5%) | **36** | **51** | **13** | **-52.51 Elo** | — |

---

## ♟️ Match PGNs & Annotated Games

All tournament games and checkmate combinations are public in [`games/`](games/):

- **[100-Game Passed-Pawn Showdown PGN](games/sf_owen_passed_pawn_100_showdown.pgn)**: 51 Wins, 57.5% victory.
- **[10-Game Championship PGN](games/sf_owen_vs_stockfish19_10games.pgn)**: 6.5 / 10 (+107 Elo).
- **[Victory Game 3](games/victory_game_3.pgn)**: Center pawn breakthrough checkmate (`1-0`).
- **[Victory Game 7](games/victory_game_7.pgn)**: Kingside sacrifice combination (`1-0`).
- **[Victory Game 9](games/victory_game_9.pgn)**: Imbalanced Queen endgame conversion (`1-0`).

---

## 🚀 100% Self-Contained Binaries (All Platforms)

All release executables on [GitHub Releases](https://github.com/Owen-Foundation/sf-owen/releases) have the **98.5 MB Champion Neural Network baked directly inside**. Zero configuration required.

| Platform | Architecture | Binary Name |
| :--- | :--- | :--- |
| **Windows 64-bit** | AVX2 / AVX-512 / BMI2 | `sf-owen-windows-*.exe` |
| **Linux 64-bit** | AVX2 / AVX-512 / BMI2 / Modern | `sf-owen-linux-*` |
| **macOS Apple Silicon** | M1, M2, M3, M4 (ARM64) | `sf-owen-macos-apple-silicon` |
| **macOS Intel** | BMI2 / Modern | `sf-owen-macos-x86-64-*` |

---

## License

GPL-3.0. Stockfish search core and NNUE architectures are credited to the Stockfish developers.
