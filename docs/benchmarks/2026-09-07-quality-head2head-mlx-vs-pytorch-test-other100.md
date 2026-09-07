# MLX vs PyTorch Quality Head-to-Head (test-other, n=100)

- model: `Qwen/Qwen3-ASR-0.6B`
- samples: `100`
- MLX WER: `0.0430`
- PyTorch WER: `0.0441`
- Delta WER (MLX-Ref): `-0.0011`
- MLX CER: `0.0211`
- PyTorch CER: `0.0214`
- Delta CER (MLX-Ref): `-0.0004`

| System | WER | CER | Mean latency (s) |
|---|---:|---:|---:|
| MLX | 0.0430 | 0.0211 | 0.3972 |
| PyTorch ref | 0.0441 | 0.0214 | 2.7631 |
