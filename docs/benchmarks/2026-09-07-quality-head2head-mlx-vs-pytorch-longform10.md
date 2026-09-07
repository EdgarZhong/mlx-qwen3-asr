# MLX vs PyTorch Quality Head-to-Head (manifest-quality-v1, n=10)

- model: `Qwen/Qwen3-ASR-0.6B`
- samples: `10`
- reference_max_chunk_sec: `0.0`
- MLX primary: `0.1059`
- PyTorch primary: `0.1799`
- Delta primary (MLX-Ref): `-0.0740`

| System | Primary | WER | CER | Mean latency (s) |
|---|---:|---:|---:|---:|
| MLX | 0.1059 | 0.1512 | 0.0600 | 3.7536 |
| PyTorch ref | 0.1799 | 0.2431 | 0.1197 | 26.7616 |
