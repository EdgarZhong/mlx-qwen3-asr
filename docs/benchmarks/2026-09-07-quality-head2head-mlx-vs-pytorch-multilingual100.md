# MLX vs PyTorch Quality Head-to-Head (manifest-quality-v1, n=100)

- model: `Qwen/Qwen3-ASR-0.6B`
- samples: `100`
- reference_max_chunk_sec: `0.0`
- MLX primary: `0.0954`
- PyTorch primary: `0.1034`
- Delta primary (MLX-Ref): `-0.0081`

| System | Primary | WER | CER | Mean latency (s) |
|---|---:|---:|---:|---:|
| MLX | 0.0954 | 0.1600 | 0.0543 | 0.6524 |
| PyTorch ref | 0.1034 | 0.1669 | 0.0564 | 12.9075 |
