from __future__ import annotations
import io, random
from pathlib import Path
import numpy as np
import torch
from ruamel.yaml import YAML
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from src.models.marl.mlp_policy_sa_lstm import MlpPolicySALSTM


def set_up_cfg_yaml(config):
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    config_stream = io.StringIO("")
    yaml.dump(config, config_stream)
    config_stream.seek(0)
    config_string = config_stream.read()
    config_stream.close()
    return config_string


def load_teacher(weight_dir: Path, device: torch.device, config: dict = None, env=None):
    saved_variables = torch.load(str(weight_dir), map_location=device)
    if config["main"]["policy_net"] in ["MultiHead_attention_LSTM", "MLP_random_LSTM"]:
        policy_kwargs = dict(
            n_lstm_layers=config["lstm_policy"]["n_lstm_layers"],
            lstm_hidden_size=config["lstm_policy"]["lstm_hidden_size"],
            enable_critic_lstm=config["lstm_policy"]["enable_critic_lstm"],
            shared_lstm=config["lstm_policy"]["shared_lstm"]
            if "shared_lstm" in config["lstm_policy"]
            else False,
        )
        saved_variables["data"] = {**saved_variables["data"], **policy_kwargs}
        policy = MlpLstmPolicy(**saved_variables["data"])
    elif config["main"]["policy_net"] == "SingleAgent_LSTM":
        obs_dim_fixed = (
            getattr(env, "obs_dim_single_fixed", 36) if env is not None else 36
        )
        policy = MlpPolicySALSTM(**saved_variables["data"], obs_dim_fixed=obs_dim_fixed)
    else:
        raise ValueError("Unknown policy net: {0}".format(config["main"]["policy_net"]))
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    policy.load_state_dict(saved_variables["state_dict"], strict=True)
    policy.to(device)
    return policy


def configure_seed(seed: int, env=None) -> None:
    print("[SEED] Set seed to {0}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.seed(seed)
