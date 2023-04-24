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
    is_score_precond: bool = True
    std_trick: bool = True
    residual_trick: bool = True
    weighted: bool = True


@dataclasses.dataclass
class OptimizationConfig:
    batch_size: int = 32
    num_epochs: int = 100
    num_warmup_epochs: int = 20
    lr: float = 1e-3
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
    num_samples_in_epoch: int = int(2**10)


@dataclasses.dataclass
class Config:
    seed: int = 42
    mode: str = "smoketest"
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
