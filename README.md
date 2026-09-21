# Qwen-Image-2512 LoRA training benchmark

Times one LoRA fine-tune of Qwen-Image-2512 with ai-toolkit on a single RTX Pro 6000 Blackwell (Workstation Edition, 96 GB). The job matches a public [r/StableDiffusion run](https://www.reddit.com/r/StableDiffusion/comments/1qg4w9z/til_renting_an_rtx_pro_6000_blackwell_can_be/): batch size 2, 8 sample images every 250 steps. That run was 8,000 steps; this one trains 1,500 by default and extrapolates to 8,000 from the steady-state rate.

## Files

- `config.yaml` is the ai-toolkit job. It uses ai-toolkit's UI defaults for a new Qwen-Image-2512 job (commit `ef8110c`), except for four changes:
  - batch size 2
  - 1,500 steps by default (`STEPS=8000 ./run.sh` for the full length)
  - 8 sample prompts
  - the model kept in bf16, with no quantization and no layer offloading
- `run.sh` sets up the environment, trains, and summarizes. It pins ai-toolkit to `ef8110c` and PyTorch to 2.13.0 (`cu130`).
- `download.py` fetches the model and the dataset.
- `summarize.py` turns `results/` into `summary.txt`.
- `results/` is written by the run and committed:
  - `timeline.log`: UTC timestamps for setup and training
  - `env.txt`: ai-toolkit commit, PyTorch/CUDA, GPU, driver, OS
  - `gpu.csv`: power, memory and utilization every 5 s
  - `checkpoints.csv`: each 250-step checkpoint and when it was written
  - `train.log.gz`: full trainer output
  - `summary.txt`: setup minutes, training hours, steady-state steps/hour, fastest and slowest 250-step block, one-off startup time, extrapolated 8,000-step hours, average power, peak GPU memory, peak temperature
- `work/` holds the clone, venv, model weights (~58 GB) and outputs. It's disposable and git-ignored.

## What is measured

- **Setup time:** clone, installs, and model and dataset download. A stopped Enverge instance keeps nothing, so this cost recurs every session. Hugging Face rate-limits anonymous downloads, so export `HF_TOKEN` before running for a faster setup. `env.txt` records whether a token was set, since it changes this number.
- **Training wall-clock:** from `run.py` start to exit, including model load and sampling.
- **Steady-state steps/hour:** from the first to the last checkpoint. This excludes model load and includes sampling every 250 steps, the same way the Reddit run was timed.
- **Extrapolated 8,000-step time:** one-off startup (model load and first sample) plus 8,000 steps at the steady-state rate. It's valid because every 250-step block does the same work (same batch, buckets, and sampling), so time per block is flat once the model is loaded. The fastest and slowest block, and peak temperature (for throttling), show whether that held.
- **GPU power, peak memory and temperature:** peak memory shows how much a 24 GB card would have to offload.

The dataset is [`diffusers/dog-example`](https://huggingface.co/datasets/diffusers/dog-example) (5 photos, one caption). Image content doesn't affect speed; resolution buckets and batch size do. At 8,000 steps the LoRA overfits, which doesn't matter for timing.

## Run it on Enverge

1. Launch an **RTX Pro 6000** instance at [app.enverge.ai](https://app.enverge.ai). Check that its SSH hostname contains the host you want, e.g. `<token>.rtx6000-0003.ssh.enverge.dev`.
2. Copy this repo to the instance, from the repo root on your machine:
   ```
   scp -r . user@<token>.rtx6000-0003.ssh.enverge.dev:~/qwen-image-2512-lora
   ```
3. Start the run detached, so it survives an SSH disconnect:
   ```
   ssh user@<token>.rtx6000-0003.ssh.enverge.dev "cd ~/qwen-image-2512-lora && HF_TOKEN=<token> nohup ./run.sh > run.out 2>&1 &"
   ```
   `HF_TOKEN=<token>` is optional; drop it to download anonymously. The machine needs Python's venv package and headers (Triton compiles a small CUDA helper against them on first use). Enverge instances ship the headers; on a bare Ubuntu 24.04 host, install both first: `sudo apt-get install -y python3.12-venv python3.12-dev`. Then re-run step 3.
4. Watch the first 20 minutes. An out-of-memory error shows up early, and the first checkpoint appears after 250 steps:
   ```
   ssh user@<token>.rtx6000-0003.ssh.enverge.dev "tail -n 5 ~/qwen-image-2512-lora/run.out; ls ~/qwen-image-2512-lora/work/output/bench/"
   ```
   If it runs out of memory, follow the comment in `config.yaml` to switch on float8 quantization, note that in `results/`, and re-run step 3. Setup is skipped the second time, and `summary.txt` keeps the first setup's timing.
5. After about 7.5 hours, check for `summary.txt`:
   ```
   ssh user@<token>.rtx6000-0003.ssh.enverge.dev "cat ~/qwen-image-2512-lora/results/summary.txt"
   ```
6. **Copy the results back before stopping the instance**, because nothing survives a stop:
   ```
   scp -r user@<token>.rtx6000-0003.ssh.enverge.dev:~/qwen-image-2512-lora/results .
   ```
7. Delete the instance from the dashboard.

For a quick check of the pipeline before the full run, prefix step 3's command with `STEPS=500`, then delete `results/` before starting the real run.
