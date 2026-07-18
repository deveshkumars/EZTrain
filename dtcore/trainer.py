import os
import sys
from datetime import datetime
import functools
from typing import Callable, Dict, Tuple, Any

# env config: EGL is Linux-only (headless clusters); macOS uses mujoco's default GL
if sys.platform == 'linux':
    os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
# persistent XLA compile cache: mjx graphs take minutes (CPU: tens of minutes) to compile
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', os.path.expanduser('~/.cache/dronetrain-xla'))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '10')

import pickle
import numpy as np

import jax
from jax import numpy as jnp
from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model as brax_model_io
from matplotlib import pyplot as plt
import mediapy as media

from .simple_env import register_env, make_camera


def build_env(env_name: str = 'simple', env_kwargs: Dict[str, Any] | None = None):
    register_env()
    return envs.get_environment(env_name, **(env_kwargs or {}))


def make_policy_inference_fn(env_name: str = 'simple',
                             env_kwargs: Dict[str, Any] | None = None,
                             normalize_observations: bool = True) -> Callable:
    """Builds a make_inference_fn for saved params without running training.

    Network config must match training time (brax PPO defaults + observation
    normalization, as used in default_train_config).
    """
    env = build_env(env_name, env_kwargs=env_kwargs)
    normalize = running_statistics.normalize if normalize_observations else (lambda x, y: x)
    network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size, preprocess_observations_fn=normalize)
    return ppo_networks.make_inference_fn(network)


def default_train_config() -> Dict[str, Any]:
    return {
        'num_timesteps': 1_000_000,
        'num_evals': 5,
        'reward_scaling': 0.1,
        'episode_length': 200,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': 10,
        'num_minibatches': 24,
        'num_updates_per_batch': 8,
        'discounting': 0.97,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-3,
        'num_envs': 512,
        'batch_size': 512,
        'seed': 0,
    }


def train(env_name: str = 'simple', config_overrides: Dict[str, Any] | None = None,
          model_dir: str = 'models/mjx_brax_policy',
          progress_callback: Callable[[int, Dict[str, float]], None] | None = None,
          env_kwargs: Dict[str, Any] | None = None,
          ) -> Tuple[Callable, Dict, Dict[str, Any]]:
    """Trains PPO on the given env and saves params to model_dir.

    Returns (make_inference_fn, params, logs)
    """
    env = build_env(env_name, env_kwargs=env_kwargs)

    cfg = default_train_config()
    if config_overrides:
        cfg.update(config_overrides)

    progress_x, progress_y, progress_yerr, times = [], [], [], [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        if progress_callback:
            progress_callback(num_steps, metrics)
        progress_x.append(num_steps)
        progress_y.append(metrics['eval/episode_reward'])
        progress_yerr.append(metrics['eval/episode_reward_std'])

    train_fn = functools.partial(ppo.train, **cfg)
    make_inference_fn, params, logs = train_fn(environment=env, progress_fn=progress)

    # persist (Brax format)
    os.makedirs(model_dir, exist_ok=True)
    brax_model_io.save_params(os.path.join(model_dir, 'policy'), params)

    # also persist as a .pkl for the C code generator, which expects the raw
    # (normalizer_state, policy_params, value_params) tuple with numpy leaves
    pkl_path = os.path.join(model_dir, 'params.pkl')

    def to_numpy(x):
        try:
            return np.asarray(x)
        except Exception:
            return x

    cpu_state = jax.tree_util.tree_map(to_numpy, params)
    with open(pkl_path, 'wb') as f:
        pickle.dump(cpu_state, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save a training curve plot
    plt.figure()
    plt.plot(progress_x, progress_y, label='Episode Reward')
    plt.xlabel('Number of Steps')
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward vs. Number of Steps')
    plt.legend()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/train.png')

    return make_inference_fn, params, logs


def evaluate(env_name: str, make_inference_fn: Callable, params: Dict,
             n_steps: int = 200, video_path: str | None = 'gifs/simple_train.mp4',
             env_kwargs: Dict[str, Any] | None = None) -> None:
    env = build_env(env_name, env_kwargs=env_kwargs)
    cam = make_camera()

    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

    if video_path:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        media.write_video(path=video_path, images=env.render(rollout, camera=cam), fps=1.0 / env.dt) 