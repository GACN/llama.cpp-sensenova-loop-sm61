# llama.cpp + SenseNova-Loop, built for Pascal (sm_61)

Unofficial Windows x64 build of [llama.cpp](https://github.com/ggml-org/llama.cpp) that can load
**SenseNova-Loop** (`sensenova_loop`) models, compiled with **CUDA 12.8 for compute capability 6.1**
(GTX 10-series, Titan X / Xp — cards that CUDA 13 dropped).

The architecture support comes from [PR #32588](https://github.com/ggml-org/llama.cpp/pull/32588)
by [blueswhen](https://github.com/blueswhen), an official contribution from the SenseNova team.
The PR is not merged into a release yet, so upstream binaries still answer:

```
error loading model: unknown model architecture: 'sensenova_loop'
```

This build exists so Pascal owners can run the model today.

---

## Measured on real hardware

| | |
|---|---|
| GPU | NVIDIA TITAN X (Pascal), 12 GB, sm_61, 480 GB/s, no tensor cores |
| CPU | Xeon E5-2698B v3, 32 GB RAM |
| OS | Windows 10 19044, driver 582.78 |
| Model | SenseNova-Loop-9B Q4_K_M (5.42 GB) |
| Context | 4096 → 6336 MiB VRAM · 8192 + MTP → 6856 MiB |
| Model load | 58 s (cold, from SSD) |

**Speed** — 600-token Chinese explanation, `-ngl 99`, `-c 8192`:

| Mode | Generation | Draft acceptance | vs plain |
|---|---|---|---|
| plain | 15.8 tok/s | — | — |
| `--spec-type draft-mtp --spec-draft-n-max 1` | 18.2 tok/s | 75 % | +15 % |
| **`--spec-type draft-mtp --spec-draft-n-max 2`** | **18.9 tok/s** | 64 % | **+19 %** |
| `--spec-draft-n-max 3` (value used in the PR example) | 16.5 tok/s | 48 % | +4 % |
| `--spec-draft-n-max 4` | 14.2 tok/s | 33 % | −10 % |

**`n-max 2` is the sweet spot on Pascal** — the PR's example value of 3 costs almost all of the gain.
Prompt processing is 326–503 tok/s depending on batch (`llama-bench -p 64` → 326 tok/s, `-p 512` → 503 tok/s).

Nothing else helps: `-fa on/off`, `-b 2048 -ub 512`, `GGML_CUDA_FORCE_MMQ=1` and
`GGML_CUDA_DISABLE_GRAPHS=1` all land within ±5 % (noise). The model is GPU-bound, not CPU-bound —
CPU stays at 14 %. Run-to-run variance is roughly ±10 %, so compare configurations in one session.

**Why it is not faster on Pascal.** The 9B checkpoint stores **32 physical trunk layers but evaluates
56 logical layers per token** (looped transformer, weights shared, per-logical-layer KV/conv state).
Bytes read per token ≈ 5.42 GB × 56/32 ≈ 9.5 GB, so the bandwidth ceiling on a 480 GB/s card is
≈ 50 tok/s. With tuned MTP we reach **18.9 tok/s (38 % of that ceiling)**; the remaining gap is Q4_K
dequantization on Pascal, which has no tensor cores, and PR #32588 adds no CUDA-specific kernels for
the loop path. The [`sensenovaloop_awq` branch of LightLLM](https://github.com/ModelTC/LightLLM)
fuses the mHC ops with Triton kernels and gets much closer to the roofline, but it is Linux + AWQ
only and does not run on Pascal.

**Quality** — same 8-task ground-truth suite used to evaluate other local models
(code unit tests, math, Chinese fact, JSON, tool call, needle-in-haystack):

| Model | Score | Time |
|---|---|---|
| **SenseNova-Loop-9B Q4_K_M** (this build) | **8/8** | 54 s |
| mimo-v2.6:9b Q4_K_M (Ollama) | 7/8 | 29 s |
| ornith-1.5:9b (Ollama) | 6/8 | 38 s |

So: roughly 3× slower than a plain 9B, but the strongest local model tested so far —
it was the only one to pass all eight tasks (the others failed long-context retrieval / palindrome
logic / tool-call JSON respectively).

---

## Usage

```
llama-server.exe -m SenseNova-Loop-9B-Q4_K_M.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 8090
```

Faster variant with the built-in MTP predictor — **use `-n-max 2`, not the 3 from the PR example**:

```
llama-server.exe -m SenseNova-Loop-9B-Q4_K_M.gguf -ngl 99 -c 8192 ^
  --spec-type draft-mtp --spec-draft-n-max 2 --no-spec-draft-backend-sampling ^
  --host 127.0.0.1 --port 8090
```

Then use the OpenAI-compatible API:

```powershell
$body = @{ messages = @(@{ role="user"; content="你好" }); max_tokens=256 } |
        ConvertTo-Json -Depth 5 -Compress
Invoke-RestMethod http://127.0.0.1:8090/v1/chat/completions -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

> On PowerShell 5.1, send the body as UTF-8 **bytes** (`[Text.Encoding]::UTF8.GetBytes(...)`).
> Passing a string makes PowerShell use the local ANSI code page and Chinese prompts get mangled.

## Required DLLs (not included)

`cublas64_12.dll` (108 MB) and `cublasLt64_12.dll` (643 MB) are too large to ship here and are
redistributable only as part of the CUDA runtime. Get them one of these ways:

1. Install the CUDA 12.x runtime — https://developer.nvidia.com/cuda-downloads (choose 12.x, **not 13.x**
   if you are on Pascal).
2. Or copy them out of any existing CUDA 12 install (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`).
3. Or copy them from an official llama.cpp `cudart-llama-bin-win-cuda-12.x-x64.zip` release asset.

Put `cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll` next to `llama-server.exe`,
or add your CUDA `bin` folder to `PATH`.

## What is inside

- `llama-server.exe`, `llama-cli.exe`, `llama-bench.exe` (+ `-impl.dll` companions)
- `ggml-cuda.dll` — 36 MB, built for `sm_61` only
- `ggml-base.dll`, `ggml-cpu.dll` (AVX2/FMA/F16C), `ggml.dll`, `llama.dll`, `mtmd.dll`
- `cudart64_12.dll`
- `BUILD-INFO.txt`

## Build recipe

```powershell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch origin pull/32588/head:sensenova-loop
git checkout sensenova-loop

# MSVC 19.44 + CUDA 12.8, Ninja generator (works with a portable MSVC, no VS install needed)
cmake -B build-cuda -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=61 `
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-cuda -j 8
```

Binary lands in `build-cuda\bin\`.

## Notes / caveats

- `sm_61` only. On a newer card this DLL will not load — rebuild with your own
  `-DCMAKE_CUDA_ARCHITECTURES`.
- The GGUF contains an MTP predictor (`blk.56.*` tensors). Without `--spec-type draft-mtp`
  llama.cpp prints those tensors as "unused ... ignoring" — that warning is expected.
- `-ngl 99` fits Q4_K_M (5.42 GB) comfortably in 12 GB.
- Image input needs a separate `mmproj-*.gguf` and `llama-mtmd-cli.exe` (built but not shipped
  in the trimmed archive; rebuild to get it).

## Credits and license

- llama.cpp — MIT, https://github.com/ggml-org/llama.cpp
- SenseNova-Loop support — PR #32588, blueswhen / SenseNova team
- SenseNova-Loop-9B weights — Apache-2.0, https://huggingface.co/sensenova/SenseNova-Loop-9B

This repository contains only build artefacts of MIT-licensed code. **No model weights are included.**
