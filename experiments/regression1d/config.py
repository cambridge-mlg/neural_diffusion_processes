from typing import ClassVar, Dict, Protocol, Mapping, Optional
import dataclasses

# from ml_collections import config_dict

@dataclasses.dataclass
class DataConfig:
    seed: int = 0
    kernel: str = "rbf"
    num_samples: int = 10_000
    num_points: int = 100
    hyperparameters: Mapping[str, float] = dataclasses.field(
        default_factory=lambda: {
            "variance": 1.0,
            "lengthscale": 0.2,
        }
    )

    seed_test: int = 1
    num_samples_test: int = 64


@dataclasses.dataclass
class SdeConfig:
    limiting_kernel: str = "white"
    limiting_kernel_hyperparameters: Mapping[str, float] = dataclasses.field(
        default_factory=lambda: {
            "variance": 1.0,
        },
    )


@dataclasses.dataclass
class OptimizationConfig:
    batch_size: int = 16
    num_steps: int = 100_000


@dataclasses.dataclass
class NetworkConfig:
    num_bidim_attention_layers: int = 2
    hidden_dim: int = 16
    num_heads: int = 4


@dataclasses.dataclass
class Config:
    seed: int = 42
    data: DataConfig = DataConfig()
    sde: SdeConfig = SdeConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    network: NetworkConfig = NetworkConfig()



toy_config = Config(
    seed=666,
    data=DataConfig(
        num_samples=1_000,
    ),
)


if __name__ == "__main__":
    from ml_collections import config_dict
    c = config_dict.ConfigDict(initial_dictionary=dataclasses.asdict(toy_config))

    from neural_diffusion_processes.ml_tools import config_utils

    config = config_utils.to_dataclass(Config, c)
    print(config_utils.get_id(config))
