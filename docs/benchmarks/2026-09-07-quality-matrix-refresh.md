# Quality and Latency Refresh (2026-09-07, v0.4.0)

Machine: Apple M4 Pro (48 GB), macOS 26, MLX 0.30.6, Python 3.11. Latency runs
were taken on an idle machine (load average under 4); quality runs are
insensitive to load. Quantized variants were produced locally with
`scripts/convert.py --quantize` (group size 64).

Why this refresh exists: v0.4.0 fixed the encoder's float32 position-embedding
table, which had been promoting every activation to float32. fp16 inference is
now 2-3x faster than the February numbers, and the quantization speedups shrink
correspondingly because they are measured against a real fp16 baseline. Output
quality is unchanged: WER moves by at most a few words on 100-sample lanes, and
8-bit reproduces fp16 output exactly on LibriSpeech test-clean.

## Commands

```bash
python scripts/benchmark_quantization_matrix.py --model Qwen/Qwen3-ASR-0.6B --configs fp16,8:64,4:64 --benchmark-runs 5 --eval-subset test-clean --eval-samples 100 --eval-sampling speaker_round_robin --json-output docs/benchmarks/2026-09-07-quant-matrix-test-clean-speaker100.json --md-output docs/benchmarks/2026-09-07-quant-matrix-test-clean-speaker100.md
python scripts/benchmark_quantization_matrix.py ... --eval-subset test-other ... (same flags)
python scripts/build_multilingual_manifest.py --languages en_us,zh_cn,ja_jp,de_de,fr_fr,es_419,ru_ru,ar_eg,hi_in,ko_kr --samples-per-language 10 --seed 20260214 --output-manifest docs/benchmarks/2026-09-07-fleurs-multilingual-100-manifest.jsonl
python scripts/build_longform_manifest.py --input-manifest docs/benchmarks/2026-09-07-fleurs-multilingual-100-manifest.jsonl --output-manifest docs/benchmarks/2026-09-07-fleurs-longform-10x75-manifest.jsonl --target-duration-sec 75 --clips-per-language 1
for MODEL in 0.6B 1.7B; for SUBSET in test-clean test-other:
  python scripts/eval_librispeech.py --model Qwen/Qwen3-ASR-$MODEL --dtype float16 --subset $SUBSET --samples 100 --sampling speaker_round_robin --json-output docs/benchmarks/2026-09-07-librispeech-$SUBSET-100[-1p7b].json
for MODEL in 0.6B 1.7B:
  python scripts/eval_manifest_quality.py --manifest-jsonl docs/benchmarks/2026-09-07-fleurs-multilingual-100-manifest.jsonl --model Qwen/Qwen3-ASR-$MODEL --dtype float16 --json-output docs/benchmarks/2026-09-07-manifest-quality-multilingual100-[0p6b|1p7b].json
python scripts/eval_manifest_quality.py --manifest-jsonl docs/benchmarks/2026-09-07-fleurs-longform-10x75-manifest.jsonl --model Qwen/Qwen3-ASR-0.6B --dtype float16 --json-output docs/benchmarks/2026-09-07-manifest-quality-longform10-0p6b.json
for CFG in fp16 8bit-g64 4bit-g64 1p7b-fp16; for CLIP in short 10s:
  python scripts/benchmark_asr.py <clip> --model <model or quantized dir> --dtype float16 --warmup-runs 3 --runs 10 --json-output docs/benchmarks/2026-09-07-latency-$CFG-$CLIP.json
```

## English Quality (LibriSpeech, 100 speaker-balanced samples per subset)

| Model | Subset | WER | CER | Mean Latency | RTF |
|---|---|---:|---:|---:|---:|
| 0.6B | test-clean | 2.33% | 0.59% | 0.35s | 0.0393 |
| 0.6B | test-other | 4.30% | 2.11% | 0.40s | 0.0553 |
| 1.7B | test-clean | 1.94% | 0.57% | 0.77s | 0.0862 |
| 1.7B | test-other | 3.45% | 1.48% | 0.66s | 0.0914 |

## Latency (median of 10 runs, idle machine)

| Configuration | Short clip (~2.5s) | 10s clip | RTF (10s) | vs fp16 (10s) |
|---|---:|---:|---:|---:|
| 0.6B fp16 (baseline) | 0.17s | 0.30s | 0.029 | — |
| 0.6B 8-bit (g64) | 0.10s | 0.23s | 0.024 | 1.32x |
| 0.6B 4-bit (g64) | 0.09s | 0.17s | 0.018 | **1.71x** |
| 1.7B fp16 | 0.36s | 0.73s | 0.077 | 2.4x slower |

## Quantization Quality (0.6B, LibriSpeech test-clean)

| Configuration | WER | CER | WER vs fp16 | Speed vs fp16 (10s clip) |
|---|---:|---:|---:|---:|
| fp16 (baseline) | 2.33% | 0.59% | — | — |
| 8-bit (g64) | 2.33% | 0.59% | +0.00pp | 1.32x |
| 4-bit (g64) | 2.59% | 0.86% | +0.26pp | 1.71x |

## Quantization Quality (0.6B, LibriSpeech test-other)

| Configuration | WER | CER | WER vs fp16 | Speed vs fp16 (10s clip) |
|---|---:|---:|---:|---:|
| fp16 (baseline) | 4.30% | 2.11% | — | — |
| 8-bit (g64) | 4.14% | 2.06% | -0.16pp | 1.32x |
| 4-bit (g64) | 5.74% | 2.71% | +1.43pp | 1.71x |

## Multilingual Quality (FLEURS, 10 languages x 10 samples)

### 0.6B (fp16)

| Language | Samples | WER | CER | Primary | Latency |
|---|---:|---:|---:|---:|---:|
| Arabic | 10 | 21.5% | 6.8% | 21.5% | 0.54s |
| Chinese | 10 | 91.7% | 5.0% | 5.0% | 0.36s |
| English | 10 | 4.6% | 1.6% | 4.6% | 0.32s |
| French | 10 | 17.3% | 9.2% | 17.3% | 0.50s |
| German | 10 | 8.0% | 4.7% | 8.0% | 0.65s |
| Hindi | 10 | 16.7% | 9.9% | 16.7% | 1.98s |
| Japanese | 10 | 89.7% | 9.3% | 9.3% | 0.51s |
| Korean | 10 | 17.2% | 6.7% | 6.7% | 0.46s |
| Russian | 10 | 8.8% | 3.4% | 8.8% | 0.62s |
| Spanish | 10 | 3.0% | 0.6% | 3.0% | 0.59s |
| **Aggregate** | **100** | **16.0%** | **5.4%** | **9.54%** | **0.65s** |

### 1.7B (fp16)

| Language | Samples | Primary | Latency |
|---|---:|---:|---:|
| Arabic | 10 | 16.0% | 1.09s |
| Chinese | 10 | 8.5% | 0.75s |
| English | 10 | 4.2% | 0.69s |
| French | 10 | 4.1% | 1.05s |
| German | 10 | 5.8% | 1.15s |
| Hindi | 10 | 17.7% | 3.12s |
| Japanese | 10 | 3.6% | 1.08s |
| Korean | 10 | 5.3% | 0.98s |
| Russian | 10 | 5.4% | 1.20s |
| Spanish | 10 | 0.7% | 1.09s |
| **Aggregate** | **100** | **6.70%** | **1.22s** |

### 0.6B vs 1.7B

| Language | 0.6B Primary | 1.7B Primary | Delta | Latency Ratio |
|---|---:|---:|---:|---:|
| Arabic | 21.5% | 16.0% | -5.5pp | 2.01x |
| Chinese | 5.0% | 8.5% | +3.5pp | 2.08x |
| English | 4.6% | 4.2% | -0.5pp | 2.18x |
| French | 17.3% | 4.1% | -13.2pp | 2.10x |
| German | 8.0% | 5.8% | -2.2pp | 1.76x |
| Hindi | 16.7% | 17.7% | +1.0pp | 1.58x |
| Japanese | 9.3% | 3.6% | -5.8pp | 2.11x |
| Korean | 6.7% | 5.3% | -1.4pp | 2.13x |
| Russian | 8.8% | 5.4% | -3.4pp | 1.95x |
| Spanish | 3.0% | 0.7% | -2.2pp | 1.87x |
| **Overall** | **9.54%** | **6.70%** | **-2.83pp** | **1.87x** |

## Long-Form Quality (FLEURS concatenated, 78-90s per clip, 0.6B fp16)

| Language | WER | CER | Primary | Latency |
|---|---:|---:|---:|---:|
| Arabic | 22.2% | 8.1% | 22.2% | 3.9s |
| Chinese | 47.6% | 3.2% | 3.2% | 2.0s |
| English | 5.7% | 1.9% | 5.7% | 2.7s |
| French | 19.7% | 11.2% | 19.7% | 3.5s |
| German | 6.1% | 3.4% | 6.1% | 3.4s |
| Hindi | 25.0% | 13.8% | 25.0% | 8.1s |
| Japanese | 89.5% | 10.6% | 10.6% | 2.8s |
| Korean | 10.3% | 3.7% | 3.7% | 3.9s |
| Russian | 11.6% | 4.2% | 11.6% | 4.1s |
| Spanish | 4.3% | 0.4% | 4.3% | 3.2s |
| **Aggregate** | **15.1%** | **6.0%** | **10.6%** | **3.8s** |

## Comparison with the 2026-02-15 refresh

| Lane | February | This refresh |
|---|---:|---:|
| 0.6B test-clean WER | 2.29% | 2.33% |
| 0.6B test-other WER | 4.20% | 4.30% |
| 1.7B test-clean WER | 1.99% | 1.94% |
| 1.7B test-other WER | 3.45% | 3.45% |
| 0.6B multilingual primary | 9.37% | 9.54% |
| 1.7B multilingual primary | 6.70% | 6.70% |
| 0.6B fp16 10s clip | 0.83s | 0.30s |
| 0.6B 4-bit 10s clip | 0.18s | 0.17s |
| 1.7B multilingual mean latency | 4.12s | 1.22s |

The multilingual manifest was rebuilt with the original seed and contains the
same 100 samples. The long-form set is new (derived from that manifest), so its
numbers are not directly comparable with the February long-form lane.

## MLX vs PyTorch parity (added after the initial refresh)

Reference stack: `qwen-asr` (PyTorch 2.10, CPU). Commands:

```bash
python scripts/eval_manifest_head2head.py --mlx-json docs/benchmarks/2026-09-07-manifest-quality-multilingual100-0p6b.json --model Qwen/Qwen3-ASR-0.6B --json-output docs/benchmarks/2026-09-07-quality-head2head-mlx-vs-pytorch-multilingual100.json --md-output docs/benchmarks/2026-09-07-quality-head2head-mlx-vs-pytorch-multilingual100.md
python scripts/eval_reference_parity_suite.py --model Qwen/Qwen3-ASR-0.6B --subsets '' --samples-per-subset 1 --manifest-jsonl docs/benchmarks/2026-09-07-fleurs-multilingual-100-manifest.jsonl --json-output docs/benchmarks/2026-09-07-reference-parity-suite-multilingual100.json
python scripts/analyze_reference_parity_mismatches.py --input-json docs/benchmarks/2026-09-07-reference-parity-suite-multilingual100.json --json-output docs/benchmarks/2026-09-07-reference-parity-suite-multilingual100-analysis.json --md-output docs/benchmarks/2026-09-07-reference-parity-suite-multilingual100-analysis.md
python scripts/eval_librispeech_head2head.py --mlx-json docs/benchmarks/2026-09-07-librispeech-test-other-100.json --model Qwen/Qwen3-ASR-0.6B --language English --json-output docs/benchmarks/2026-09-07-quality-head2head-mlx-vs-pytorch-test-other100.json --md-output docs/benchmarks/2026-09-07-quality-head2head-mlx-vs-pytorch-test-other100.md
python scripts/eval_manifest_head2head.py --mlx-json docs/benchmarks/2026-09-07-manifest-quality-longform10-0p6b.json --model Qwen/Qwen3-ASR-0.6B --json-output docs/benchmarks/2026-09-07-quality-head2head-mlx-vs-pytorch-longform10.json --md-output docs/benchmarks/2026-09-07-quality-head2head-mlx-vs-pytorch-longform10.md
```

| Lane | MLX | PyTorch | February MLX / PyTorch |
|---|---:|---:|---:|
| Multilingual-100 primary error | 9.54% | 10.34% | 9.54% / 10.34% |
| Multilingual-100 token / text match | 66% / 68% | — | 64% / 67% |
| LibriSpeech test-other WER | 4.30% | 4.41% | 4.20% / 4.41% |
| Long-form (new set) primary error | 10.59% | 17.99% | 11.56% / 17.99% (old set) |
