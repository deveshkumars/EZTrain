import argparse
import os
import sys
import shutil
from dtcore import trainer
from dtcore.simple_env import default_model_xml_path


def generate_c_code_from_model(model_dir: str, out_dir: str) -> None:
    """Generate C code from saved params.pkl using weights_to_firmware submodule."""
    submodule_path = os.path.join(os.path.dirname(__file__), 'weights_to_firmware')
    if submodule_path not in sys.path:
        sys.path.append(submodule_path)
    try:
        import quad_gen.get_models as wt_get_models
    except Exception as e:
        raise RuntimeError(f"Failed to import weights_to_firmware (looked in: {submodule_path}). Error: {e}")

    # absolute_path=True to use out_dir exactly as given
    wt_get_models.save_result(model_dir=model_dir, out_dir=out_dir, osi=False, absolute_path=True)


def copy_model_to_weights_to_firmware(model_dir: str) -> str:
    """Copy params.pkl from model_dir to weights_to_firmware's input_model folder."""
    pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not pkl_files:
        raise RuntimeError(f"No .pkl files found in {model_dir}")

    pkl_file = pkl_files[0]
    source_path = os.path.join(model_dir, pkl_file)

    input_model_dir = os.path.join(os.path.dirname(__file__), 'weights_to_firmware', 'input_model')
    os.makedirs(input_model_dir, exist_ok=True)

    dest_path = os.path.join(input_model_dir, pkl_file)
    shutil.copy2(source_path, dest_path)
    print(f"Copied {pkl_file} to {dest_path}")

    return input_model_dir


def copy_network_evaluate_to_firmware(weights_to_firmware_dir: str) -> None:
    """Copy generated network_evaluate.c into the firmware source tree as network_evaluate.h.

    The firmware (firmware/ai_drone_firmware/rl_tools_controller.c) includes
    "network_evaluate.h" from its own directory, so the header must land there.
    """
    network_evaluate_c = os.path.join(weights_to_firmware_dir, 'output_model', 'network_evaluate.c')
    if not os.path.exists(network_evaluate_c):
        raise RuntimeError(f"network_evaluate.c not found at {network_evaluate_c}")

    firmware_dir = os.path.join(os.path.dirname(__file__), 'firmware', 'ai_drone_firmware')
    if not os.path.isdir(firmware_dir):
        raise RuntimeError(
            f"Firmware submodule not found at {firmware_dir}. "
            "Run: git submodule update --init --recursive"
        )

    dest_path = os.path.join(firmware_dir, 'network_evaluate.h')
    shutil.copy2(network_evaluate_c, dest_path)
    print(f"Copied network_evaluate.c to {dest_path}")


def main():
    parser = argparse.ArgumentParser(description='DroneTrain entrypoint')
    parser.add_argument('--env', default='simple', help='Environment name registered in Brax')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training or with loaded params')
    parser.add_argument('--model_dir', default='models/mjx_brax_policy', help='Directory to save/load model params')
    parser.add_argument('--video', default='gifs/simple_train.mp4', help='Video path for evaluation (set empty to skip)')
    parser.add_argument('--steps', type=int, default=200, help='Evaluation steps')
    parser.add_argument('--model_xml', default=None, help='Override MuJoCo XML path for the environment (defaults to submodules/Custom-Crazyflie-Mujoco-Model/scene_mjx.xml)')
    parser.add_argument('--timesteps', type=int, default=None, help='Override number of training timesteps (default: trainer config, 1M)')
    parser.add_argument('--num_envs', type=int, default=None, help='Override number of parallel envs (reduce on CPU/laptop)')
    parser.add_argument('--batch_size', type=int, default=None, help='Override PPO batch size (reduce on CPU/laptop)')
    parser.add_argument('--num_evals', type=int, default=None, help='Override number of eval checkpoints during training')
    args = parser.parse_args()

    # Use default model_xml if none specified
    model_xml_path = args.model_xml if args.model_xml else default_model_xml_path()
    env_kwargs = {'model_xml_path': model_xml_path}

    if not args.model_xml:
        print(f"Using default model XML: {model_xml_path}")

    make_inference_fn = None
    params = None

    if args.train:
        config_overrides = {}
        if args.timesteps is not None:
            config_overrides['num_timesteps'] = args.timesteps
        if args.num_envs is not None:
            config_overrides['num_envs'] = args.num_envs
        if args.batch_size is not None:
            config_overrides['batch_size'] = args.batch_size
        if args.num_evals is not None:
            config_overrides['num_evals'] = args.num_evals
        make_inference_fn, params, _ = trainer.train(env_name=args.env, model_dir=args.model_dir, env_kwargs=env_kwargs, config_overrides=config_overrides or None)
        
        # After training: params.pkl -> C code -> firmware header
        print("Copying model file to weights_to_firmware input_model folder...")
        input_model_dir = copy_model_to_weights_to_firmware(args.model_dir)

        print("Generating C code from params.pkl...")
        out_dir = os.path.join(os.path.dirname(__file__), 'weights_to_firmware', 'output_model')
        os.makedirs(out_dir, exist_ok=True)
        generate_c_code_from_model(input_model_dir, out_dir)
        print("C code generation completed.")

        weights_to_firmware_dir = os.path.join(os.path.dirname(__file__), 'weights_to_firmware')
        copy_network_evaluate_to_firmware(weights_to_firmware_dir)
        print("Complete workflow finished successfully!")
    
    if args.eval:
        if make_inference_fn is None or params is None:
            # load saved params and build a matching inference fn (no training)
            from brax.io import model as brax_model_io
            make_inference_fn = trainer.make_policy_inference_fn(env_name=args.env, env_kwargs=env_kwargs)
            params = brax_model_io.load_params(os.path.join(args.model_dir, 'policy'))
        trainer.evaluate(env_name=args.env, make_inference_fn=make_inference_fn, params=params, n_steps=args.steps, video_path=args.video or None, env_kwargs=env_kwargs)


if __name__ == "__main__":
    main()
