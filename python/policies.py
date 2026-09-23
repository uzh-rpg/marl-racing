import torch
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from src.models.marl.custom_perceiver import CustomPerceiver
from src.models.marl.mlp_policy_sa_lstm import MlpPolicySALSTM


def make_policy(cfg, env):
    if cfg["ppo"]["activation"] == "Tanh":
        mlp_activation = torch.nn.Tanh
    elif cfg["ppo"]["activation"] == "LeakyReLU":
        mlp_activation = torch.nn.LeakyReLU
    else:
        mlp_activation = torch.nn.ReLU
    if cfg["main"]["policy_net"] == "MultiHead_attention_LSTM":
        policy = MlpLstmPolicy
        encoder_kwargs = dict(
            variable_feature_dim=env.variable_feature_dim,
            obs_dim_variable=env.obs_dim_single_variable,
            obs_dim_fixed=env.obs_dim_single_fixed,
        )
        latent_dim = (
            cfg["perceiver"]["num_queries"]
            * cfg["perceiver"]["num_heads"]
            * cfg["perceiver"]["dim_head"]
            + env.obs_dim_single_fixed
        )
        policy_kwargs = dict(
            features_extractor_class=CustomPerceiver,
            features_extractor_kwargs=dict(
                features_dim=latent_dim,
                model_cfg={"perceiver": cfg["perceiver"]},
                encoder_kwargs=encoder_kwargs,
            ),
            n_lstm_layers=cfg["lstm_policy"]["n_lstm_layers"],
            lstm_hidden_size=cfg["lstm_policy"]["lstm_hidden_size"],
            enable_critic_lstm=cfg["lstm_policy"]["enable_critic_lstm"],
            shared_lstm=cfg["lstm_policy"]["shared_lstm"]
            if "shared_lstm" in cfg["lstm_policy"]
            else False,
            activation_fn=mlp_activation,
            net_arch=[dict(pi=cfg["ppo"]["net_arch_pi"], vf=cfg["ppo"]["net_arch_vf"])],
            log_std_init=cfg["ppo"]["log_std_init"],
        )
    elif cfg["main"]["policy_net"] == "SingleAgent_LSTM":
        if cfg["environment"]["num_agents"] > 1:
            raise ValueError(
                "SingleAgent_LSTM policy is only valid for single agent envs"
            )
        policy = MlpPolicySALSTM
        policy_kwargs = dict(
            activation_fn=mlp_activation,
            net_arch=[dict(pi=cfg["ppo"]["net_arch_pi"], vf=cfg["ppo"]["net_arch_vf"])],
            log_std_init=cfg["ppo"]["log_std_init"],
            obs_dim_fixed=env.obs_dim_single_fixed,
            lstm_hidden_size=cfg["lstm_policy"]["lstm_hidden_size"],
            n_lstm_layers=cfg["lstm_policy"]["n_lstm_layers"],
        )
    elif cfg["main"]["policy_net"] == "MLP_random_LSTM":
        policy = MlpLstmPolicy
        policy_kwargs = dict(
            activation_fn=mlp_activation,
            net_arch=[dict(pi=cfg["ppo"]["net_arch_pi"], vf=cfg["ppo"]["net_arch_vf"])],
            log_std_init=cfg["ppo"]["log_std_init"],
            n_lstm_layers=cfg["lstm_policy"]["n_lstm_layers"],
            lstm_hidden_size=cfg["lstm_policy"]["lstm_hidden_size"],
            enable_critic_lstm=cfg["lstm_policy"]["enable_critic_lstm"],
            shared_lstm=cfg["lstm_policy"]["shared_lstm"]
            if "shared_lstm" in cfg["lstm_policy"]
            else False,
        )
    else:
        raise NotImplementedError(
            f"Policy net {cfg['main']['policy_net']} not implemented"
        )
    return (policy, policy_kwargs)
