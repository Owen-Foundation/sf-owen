# sf-owen — Stockfish nets for the Owen engine

Use Stockfish's world-class networks inside [Owen](https://github.com/Owen-Foundation/Owen):
**load** them, **fine-tune** them further, and **merge** the support into the engine —
so Owen can stand on SF knowledge and then surpass it. GPL-3.0.

`DefenceTest` stays the home of from-scratch training; this repo is the
transfer-learning track.

## Status

- [x] Reference net vendored info: `nn-134a887f4c8f.nnue` (SF master, ~99 MB)
- [x] Architecture mapped ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))
- [ ] `loader/` — parse `.nnue` → tensors (all versions)
- [ ] `finetune/` — continue training on Owen self-play data
- [ ] Engine merge — Owen loads `.nnue` natively alongside `.o2nn`
- [ ] Gate vs `o2` nets on DefenceTest, promote on green

See [docs/PLAN.md](docs/PLAN.md) for the phased roadmap.
