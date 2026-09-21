#!/usr/bin/env bash
# Times a Qwen-Image-2512 LoRA training run with ai-toolkit on one GPU.
# Downloads and installs go under ./work (disposable); everything worth keeping lands in ./results.
set -euo pipefail
cd "$(dirname "$0")"

STEPS="${STEPS:-1500}"
# The Reddit run was 8,000 steps; summary.txt extrapolates to it from the steady-state rate.
TARGET_STEPS=8000
AI_TOOLKIT_COMMIT=ef8110cef7c36e9812619616e3309d52f69f16ea
export HF_HOME="$PWD/work/hf"
# Optional: export HF_TOKEN=<token> before running. huggingface_hub picks it up and downloads faster than anonymous requests.

mkdir -p results work
mark() { echo "$(date -u +%FT%TZ) $1" | tee -a results/timeline.log; }

mark setup_start
if [ ! -d work/ai-toolkit ]; then
  git clone https://github.com/ostris/ai-toolkit.git work/ai-toolkit
  git -C work/ai-toolkit checkout "$AI_TOOLKIT_COMMIT"
fi
if [ ! -d work/venv ]; then
  python3 -m venv work/venv
  work/venv/bin/pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
  work/venv/bin/pip install -r work/ai-toolkit/requirements.txt
fi
work/venv/bin/python download.py
for image in work/dataset/*.jpeg; do echo "a photo of sks dog" > "${image%.jpeg}.txt"; done
mark setup_end

{
  echo "ai-toolkit $(git -C work/ai-toolkit rev-parse HEAD)"
  work/venv/bin/pip show torch | grep ^Version
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader
  grep PRETTY_NAME /etc/os-release
  uname -r
  echo "steps $STEPS"
  echo "hf_token $([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"
} > results/env.txt

sed "s/STEPS_PLACEHOLDER/$STEPS/" config.yaml > work/config.yaml
nvidia-smi --query-gpu=timestamp,power.draw,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits -l 5 > results/gpu.csv &
gpu_logger=$!

mark train_start
work/venv/bin/python work/ai-toolkit/run.py work/config.yaml > results/train.log 2>&1 || mark train_failed
mark train_end
kill "$gpu_logger"

for checkpoint in work/output/bench/*.safetensors; do
  echo "$(basename "$checkpoint"),$(date -u -r "$checkpoint" +%s)"
done > results/checkpoints.csv
gzip -f results/train.log

work/venv/bin/python summarize.py "$TARGET_STEPS" "$STEPS" | tee results/summary.txt
