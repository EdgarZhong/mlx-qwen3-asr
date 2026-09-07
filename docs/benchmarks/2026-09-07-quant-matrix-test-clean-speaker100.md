# Quantization Matrix

- Source model: `Qwen/Qwen3-ASR-0.6B`
- Eval subset: `test-clean`
- Eval samples: `100`
- Eval sampling: `speaker_round_robin`
- Benchmark runs: `5`
- Long clip length: `10s`

| Config | Short Mean (s) | Short RTF | Long Mean (s) | Long RTF | WER | CER | Eval RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.2034 | 0.0803 | 0.4155 | 0.0416 | 0.023306 | 0.005865 | 0.0529 |
| 8bit-g64 | 0.1023 | 0.0404 | 0.1760 | 0.0176 | 0.023306 | 0.005865 | 0.0332 |
| 4bit-g64 | 0.0840 | 0.0331 | 0.1520 | 0.0152 | 0.025896 | 0.008608 | 0.0279 |

## Relative to fp16

- `8bit-g64` long-clip speedup vs fp16: `2.36x` (WER delta `+0.000000`)
- `4bit-g64` long-clip speedup vs fp16: `2.73x` (WER delta `+0.002590`)
