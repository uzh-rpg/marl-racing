import argparse
from pathlib import Path
import torch
import yaml
from flightgym import MarlNoCameraVecEnv_v0
from stable_baselines3.common.utils import get_linear_fn
from policies import make_policy
from env import MarlFlightEnvVec
from src.models.marl.ppo_recurrent import RecurrentPPO
from src.utils.train_utils import configure_seed, set_up_cfg_yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "sa": "SA/config.yaml",
    "ippo": "IPPO/config.yaml",
    "sp": "SP/config.yaml",
    "lp": "LP/config.yaml",
    "lp_mlp": "LP-MLP/config.yaml",
}


def run(args):
    cfg = yaml.safe_load((ROOT / f"checkpoints/{CONFIGS[args.method]}").read_text())
    cfg["environment"]["tracks"] = str(ROOT / "track.yaml")
    cfg["main"]["test_env"] = "no"
    if args.num_envs:
        cfg["main"]["num_envs"] = args.num_envs
    if args.threads:
        cfg["main"]["num_threads"] = args.threads
    if args.steps:
        cfg["ppo"]["n_steps"] = args.steps
        cfg["ppo"]["batch_size"] = cfg["main"]["num_envs"] * args.steps
    if args.epochs:
        cfg["ppo"]["n_epochs"] = args.epochs
    for path in cfg["environment"].get("opponents_pool") or []:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(ROOT / path)

    def make_env(config):
        return MarlFlightEnvVec(
            MarlNoCameraVecEnv_v0(set_up_cfg_yaml(config)),
            num_agents=config["environment"]["num_agents"],
            num_selfplay_agents=config["environment"]["num_selfplay_agents"],
        )

    env = make_env(cfg)
    configure_seed(args.seed, env=env)
    (policy, policy_kwargs) = make_policy(cfg, env)
    algo = RecurrentPPO
    keys = (
        "gae_lambda",
        "gamma",
        "n_steps",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "batch_size",
        "n_epochs",
        "clip_range",
        "use_sde",
        "sde_sample_freq",
    )
    lr = cfg["ppo"]["learning_rate"]
    model = algo(
        policy=policy,
        policy_kwargs=policy_kwargs,
        env=env,
        env_cfg=cfg,
        use_tanh_act=True,
        tensorboard_log=str(Path(args.output).resolve()),
        learning_rate=get_linear_fn(
            float(lr["start"]), float(lr["end"]), float(lr["end_fraction"])
        ),
        device=args.device,
        verbose=1,
        **{key: cfg["ppo"][key] for key in keys},
    )
    model.learn(
        total_timesteps=args.timesteps,
        log_interval=(10, 100, 10),
        env_cfg=cfg,
        tb_log_name=CONFIGS[args.method].split("/")[0],
    )
    folder = Path(model.logger.get_dir()) / "final"
    folder.mkdir(exist_ok=True)
    for agent in range(model.num_active_agents):
        model.policy[agent].save(str(folder / f"{agent + 1}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=CONFIGS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=200000000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epochs", type=int)
    run(parser.parse_args())
