# Stockfish master network architecture (reference)

Source: `official-stockfish/Stockfish@master`, `src/nnue/nnue_architecture.h`.
Documented here so the loader and fine-tuner implement exactly this.

## Feature sets

| Set | Type | Notes |
|-----|------|-------|
| PSQ | `HalfKAv2_hm` | Half-King All-triples, horizontal mirror |
| Threat | `FullThreats` | Every piece-square threatened flag |
| Pair | `PP_3Wide` | 3-wide piece-pair features |

`PSQTBuckets = 8`, `LayerStacks = 8` (king-bucket dependent layer stacks).

## Layers

```
L1 = 1024, L2 = 32, L3 = 32
fc_0 : AffineTransformSparseInput<1024, 32>
ac    : SqrClippedReLU<32> + ClippedReLU<32>   (concatenated paths)
fc_1 : AffineTransform<64, 32>
ac    : SqrClippedReLU<32> + ClippedReLU<32>
fc_2 : AffineTransform<64 + 64, 1>             (FC_0x2 ++ FC_1x2 -> value)
```

Key differences from Owen's `.o2nn` (HalfKP+Threat, 1024→16→32→1+policy,
plain clamps): different feature layout, bucketed stacks, squared-clipped
activations, deeper head. The loader must implement all of the above
bit-exactly; verified by matching Stockfish's own eval output on test FENs.

## File container

Standard `.nnue`: magic + version + hash + arch-hash header, then
little-endian int16/int32/int8 tensors in layer order.
