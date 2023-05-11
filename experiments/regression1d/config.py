import dataclasses


@dataclasses.dataclass
class DataConfig:
    seed: int = 0
    dataset: str = "se"
    num_samples_in_epoch: int = int(2**14)


@dataclasses.dataclass
class SdeConfig:
    limiting_kernel: str = "noisy-se"
    limiting_kernel_noise_variance: float = 0.05
    t0: float = 5e-4
    score_parametrization: str = "preconditioned_k"
    std_trick: bool = True
    residual_trick: bool = True
    loss: str = "l2"
    exact_score: bool = False

    def __post_init__(self):
        assert self.score_parametrization.lower() in [
            "preconditioned_s", "preconditioned_k", "none", "y0",
        ], "Unknown score parametrization {}.".format(self.score_parametrization)

        assert self.loss in ["l1", "l2"], "Unknown loss {}.".format(self.loss)


@dataclasses.dataclass
class OptimizationConfig:
    batch_size: int = 256
    num_epochs: int = 300
    num_warmup_epochs: int = 10
    num_decay_epochs: int = 200
    init_lr: float = 1e-4
    peak_lr: float = 1e-3
    end_lr: float = 1e-5
    ema_rate: float = 0.999

    def __post_init__(self):
        assert self.num_epochs > self.num_warmup_epochs


@dataclasses.dataclass
class NetworkConfig:
    num_bidim_attention_layers: int = 5
    hidden_dim: int = 64
    num_heads: int = 8
    translation_invariant: bool = True


@dataclasses.dataclass
class EvalConfig:
    batch_size: int = 32
    num_samples_in_epoch: int = int(2**7)


@dataclasses.dataclass
class Config:
    seed: int = 42
    mode: str = "train"
    eval: EvalConfig = EvalConfig()
    data: DataConfig = DataConfig()
    sde: SdeConfig = SdeConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    network: NetworkConfig = NetworkConfig()
    experiment_dir: str = ""



toy_config = Config(
    seed=666,
    data=DataConfig(
        num_samples_in_epoch=32
    ),
)


if __name__ == "__main__":
    from ml_collections import config_dict
    c = config_dict.ConfigDict(initial_dictionary=dataclasses.asdict(toy_config))

    from neural_diffusion_processes.ml_tools import config_utils

    config = config_utils.to_dataclass(Config, c)
    print(config_utils.get_id(config))
