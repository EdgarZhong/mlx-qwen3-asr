# Quantization Matrix

- Source model: `Qwen/Qwen3-ASR-0.6B`
- Eval subset: `test-other`
- Eval samples: `100`
- Eval sampling: `speaker_round_robin`
- Benchmark runs: `5`
- Long clip length: `10s`

| Config | Short Mean (s) | Short RTF | Long Mean (s) | Long RTF | WER | CER | Eval RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.1372 | 0.0542 | 0.2850 | 0.0285 | 0.043016 | 0.021071 | 0.0493 |
| 8bit-g64 | 0.1017 | 0.0401 | 0.2267 | 0.0227 | 0.041423 | 0.020581 | 0.0335 |
| 4bit-g64 | 0.0841 | 0.0332 | 0.2192 | 0.0219 | 0.057355 | 0.027073 | 0.0270 |

## Relative to fp16

- `8bit-g64` long-clip speedup vs fp16: `1.26x` (WER delta `-0.001593`)
- `4bit-g64` long-clip speedup vs fp16: `1.30x` (WER delta `+0.014339`)
