# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Trains a Crazyflie 2.0 hover policy (PPO via brax on MuJoCo MJX physics) and automatically converts the trained MLP into dependency-free C code compiled into Crazyflie firmware. One command runs the whole chain:

train → `models/<dir>/{policy,params.pkl}` → copy to `weights_to_firmware/input_model/` → generate `output_model/network_evaluate.c` → install as `firmware/ai_drone_firmware/network_evaluate.h`.

## Commands

```bash
uv sync                                   # set up env (Python 3.11 via .python-version)
git submodule update --init --recursive   # required once; ai_drone_firmware needs git-lfs installed

# Full training run (sized for CUDA GPU / cluster)
uv run main.py --train --eval

# On the Oscar cluster: NEVER run training/JAX on a login node (vscode1 etc.) —
# submit to a GPU node via Slurm and poll with squeue/sacct
sbatch run_gpu_full.sh    # L40S; a 10M-timestep run takes ~3-5 min

# CPU/laptop pipeline smoke test (full training on CPU dies in XLA compile — see gotchas)
uv run main.py --train --eval --timesteps 0 --num_envs 16 --batch_size 32 --num_evals 2 --steps 50

# Evaluate existing params, write video
uv run main.py --eval --steps 200 --video gifs/simple_train.mp4
```

There are no tests or linters configured. Verification is running the pipeline and checking artifacts (`models/<dir>/params.pkl`, `weights_to_firmware/output_model/network_evaluate.c`, `firmware/ai_drone_firmware/network_evaluate.h`, `plots/train.png`, the eval video).

Firmware compile/flash (optional, needs `arm-none-eabi-gcc` + hardware): `./deploy_to_drone.sh` on the machine with the Crazyradio; details in `firmware/ai_drone_firmware/README.MD`.

### Training length matters

brax saves the *final* params, not the best. On this task reward plateaus at ~395
(ceiling ~400) by ~5M timesteps; a 20M run was observed to collapse to ~60 in its
last ~1M steps, silently shipping a broken policy through the whole firmware chain.
Use ~10M timesteps and always verify the *last* point of `plots/train.png` is still
on the plateau (and that the drone stays in frame in the eval video) before trusting
`params.pkl`.

## Architecture

- `main.py` — CLI + the post-training workflow (copy pkl, run C codegen, install header). Workflow functions live here, not in dtcore.
- `dtcore/simple_env.py` — `SimpleEnv(PipelineEnv)`: hover-at-1m task, backend `mjx`, obs = concat(qpos, qvel) (13-dim), 4 motor actions. Registered under the brax env name `simple`. Also owns `default_model_xml_path()` → `submodules/Custom-Crazyflie-Mujoco-Model/scene_mjx.xml` (which includes `cf2_mjx_low.xml`).
- `dtcore/trainer.py` — `train()` (brax PPO, saves params two ways), `evaluate()` (rollout + video via mediapy), `make_policy_inference_fn()` (rebuild inference net for saved params without training), `default_train_config()` (1M timesteps, 512 envs — GPU-sized).
- `weights_to_firmware/` (submodule) — `quad_gen.get_models.save_result(model_dir, out_dir, absolute_path=True)` expects `<model_dir>/params.pkl`, generates C via a TF1-compat session.
- `firmware/ai_drone_firmware/` (submodule) — modified crazyflie firmware; `rl_tools_controller.c` does `#include "network_evaluate.h"` from its own directory. That is the required install location for generated code.

### Cross-component invariants (break these and the pipeline silently degrades)

- `params.pkl` is a numpy-ified pickle of the brax PPO params tuple `(normalizer_state, policy_params, value_params)`; `quad_gen` takes element `[1]` as the policy and reads flax MLP layers named `hidden_0`..`hidden_4` (policy net = 4×32 hidden + output, the brax PPO default). Changing `policy_hidden_layer_sizes` beyond 5 total layers breaks the C generator.
- `trainer.train()` writes into `model_dir` as a *directory*: `policy` (brax `save_params` format, loaded by `--eval`) and `params.pkl` (consumed by codegen). Anything reading models must use those two paths.
- The env's observation normalization (`normalize_observations=True`) is baked into saved params; `make_policy_inference_fn` must keep matching it.

## Platform gotchas

- `MUJOCO_GL=egl` is Linux-only; setting it on macOS makes `import mujoco` raise. `trainer.py` sets it conditionally — keep it that way.
- `jax[cuda12]` is marked `sys_platform == 'linux'` in pyproject; macOS runs CPU jax.
- On CPU, XLA compilation of the PPO *training step* exceeds laptop memory (>27 GB observed, then SIGKILL/exit 137) — even with tiny configs and compile-taming XLA flags. Real training runs go on the Linux/CUDA cluster; locally verify with `--timesteps 0` (runs the whole pipeline minus learning iterations, in seconds). Eval, rendering, and C conversion all work locally. Never run two JAX processes concurrently. Long-running commands launched from Claude Code's Bash tool are killed at the 600s cap — launch detached (`nohup ... & disown`) and watch the log file.
- `trainer.py` sets `JAX_COMPILATION_CACHE_DIR=~/.cache/dronetrain-xla` (setdefault) so repeat compiles are fast. Delete that directory if compile artifacts seem stale.
- The repo was once copied between machines chmod-ing everything; `core.fileMode false` is set in the repo and submodules. Don't "fix" phantom mode-only diffs.
- `firmware/ai_drone_firmware` requires git-lfs (`brew install git-lfs && git lfs install`) or git operations inside it fail.
- Generated artifacts land *inside* the two submodules (`network_evaluate.h`, `input_model/`, `output_model/`). To publish a new policy: commit on each submodule's `main` (they sit on detached HEADs matching `main`) and push, then commit the pointer bumps in the parent. From Oscar the parent's SSH remote fails to auth — push with `git push https://github.com/deveshkumars/EZTrain.git master`.
