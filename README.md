# DroneTrain

Train a Crazyflie 2.0 hover policy with PPO in MuJoCo MJX (via brax), then automatically convert the trained network into C code ready for Crazyflie firmware.

```
main.py --train
   │
   ├─ 1. PPO training on SimpleEnv (MJX physics, Crazyflie model from submodule)
   │       └─ saves models/<dir>/policy (brax) + models/<dir>/params.pkl (numpy)
   ├─ 2. params.pkl → weights_to_firmware/input_model/
   ├─ 3. C code generation (weights_to_firmware) → output_model/network_evaluate.c
   └─ 4. Header install → firmware/ai_drone_firmware/network_evaluate.h
           └─ built into the Crazyflie firmware (see firmware README)
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.11 is pulled automatically via `.python-version`)
- `git-lfs` (needed by the `firmware/ai_drone_firmware` submodule) — `brew install git-lfs && git lfs install`
- `ffmpeg` on PATH for evaluation videos — `brew install ffmpeg` (macOS) or `module load ffmpeg` (cluster)
- Optional, for building/flashing firmware: `arm-none-eabi-gcc` toolchain and a Crazyflie 2.x

## Setup

```bash
git clone --recurse-submodules https://github.com/deveshkumars/EZTrain
cd EZTrain
uv sync
```

If you cloned without submodules: `git submodule update --init --recursive`

GPU (CUDA 12) jax is installed automatically on Linux; macOS gets CPU jax.

## Usage

Train then evaluate (full run — sized for a CUDA GPU):

```bash
uv run main.py --train --eval
```

On a Slurm cluster (e.g. Oscar), never train on a login node — submit the job
to a GPU node instead:

```bash
sbatch run_gpu_full.sh   # L40S, ~3–5 min for a 10M-step run
```

The default config trains 1M timesteps; the reward for the hover task keeps
climbing well past that. `--timesteps 10000000` reaches the reward plateau
(~395 of a ~400 ceiling). Longer is not better: a 20M-step run was observed to
collapse in its final ~1M steps, and brax saves the *final* weights, not the
best — always check that the last point of `plots/train.png` is still on the
plateau before trusting the saved policy.

Full training is designed for a CUDA GPU (Linux cluster). On a CPU-only laptop,
XLA compilation of the PPO training step needs more memory than most machines
have (observed >27 GB before the OS kills the process). You can still verify
the entire pipeline locally — training init, param saving, C code generation,
firmware header install, eval, and video — with a zero-timestep run:

```bash
uv run main.py --train --eval --timesteps 0 --num_envs 16 --batch_size 32 --num_evals 2 --steps 50
```

Evaluate previously saved params only:

```bash
uv run main.py --eval --steps 200 --video gifs/simple_train.mp4
```

CLI flags:
- `--env`: Brax environment name (we register `simple`).
- `--model_xml`: MuJoCo XML override (default: `submodules/Custom-Crazyflie-Mujoco-Model/scene_mjx.xml`, which includes `cf2_mjx_low.xml`).
- `--model_dir`: Directory for saved params (default `models/mjx_brax_policy`; contains `policy` in brax format and `params.pkl`).
- `--timesteps`, `--num_envs`, `--batch_size`, `--num_evals`: training config overrides (defaults in `dtcore/trainer.py:default_train_config`).
- `--steps`: evaluation rollout length (default 200).
- `--video`: evaluation video output path (set empty to skip).

Training also writes a reward curve to `plots/train.png`.

## Automatic workflow after training

`--train` runs the complete pipeline: after saving params it copies `params.pkl`
into `weights_to_firmware/input_model/`, generates `network_evaluate.c` (a
dependency-free C implementation of the policy MLP), and installs it as
`firmware/ai_drone_firmware/network_evaluate.h`, where the firmware's
`rl_tools_controller.c` includes it.

## Building and flashing the firmware

On the machine with the Crazyradio PA dongle attached (not the cluster):

```bash
./deploy_to_drone.sh               # preflight checks, build, prompt, flash
./deploy_to_drone.sh --build-only  # just produce build/cf2.bin
CF_URI=radio://0/80/2M ./deploy_to_drone.sh   # override radio URI
```

Needs `arm-none-eabi-gcc`, `cfloader` (`pip install cfclient`), and git-lfs.
First flight: use a net/cage and keep cfclient connected as a kill switch —
the policy is trained purely in simulation.

Manual steps (what the script does — see also `firmware/ai_drone_firmware/README.MD`):

```bash
cd firmware/ai_drone_firmware
git submodule update --init --recursive -- external/crazyflie_firmware
git submodule update --init -- external/rl_tools
cd external/crazyflie_firmware && make cf2_defconfig && cd ../..
make
cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
```

## Programmatic usage

```python
from dtcore import trainer

# Train (uses default scene_mjx.xml from submodules)
make_inference_fn, params, _ = trainer.train(
    env_name='simple',
    model_dir='models/mjx_brax_policy',
    config_overrides={'num_timesteps': 100_000},  # optional
)

# Evaluate
trainer.evaluate(
    env_name='simple',
    make_inference_fn=make_inference_fn,
    params=params,
    n_steps=200,
    video_path='gifs/simple_train.mp4',
)

# Load saved params later without retraining
from brax.io import model as brax_model_io
make_inference_fn = trainer.make_policy_inference_fn(env_name='simple')
params = brax_model_io.load_params('models/mjx_brax_policy/policy')
```

## Project structure

```
EZTrain/
├── main.py                    # CLI entrypoint + train→firmware workflow
├── deploy_to_drone.sh         # Build + flash firmware onto a Crazyflie
├── run_gpu_full.sh            # sbatch script for GPU training on the cluster
├── dtcore/
│   ├── simple_env.py          # SimpleEnv (hover at 1 m), brax env registration
│   └── trainer.py             # PPO training/eval, param saving, plotting
├── submodules/
│   └── Custom-Crazyflie-Mujoco-Model/   # MJCF models (scene_mjx.xml → cf2_mjx_low.xml)
├── weights_to_firmware/       # Submodule: params.pkl → network_evaluate.c
├── firmware/
│   └── ai_drone_firmware/     # Submodule: Crazyflie firmware consuming network_evaluate.h
├── models/                    # Trained policy params
├── plots/                     # Training curves
└── gifs/                      # Evaluation videos
```

## Platform notes

- Rendering: EGL is configured automatically on Linux (headless clusters). On
  macOS the mujoco default (CGL/GLFW) is used; do not set `MUJOCO_GL=egl` on
  macOS — it breaks `import mujoco`.
- Training is intended for GPU; on CPU the training-step compile exhausts
  laptop memory. Use the `--timesteps 0` pipeline smoke test locally instead.
  A persistent XLA compile cache is kept at `~/.cache/dronetrain-xla`.
