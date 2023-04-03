from typing import ClassVar, Dict, Protocol, Mapping, Optional

import jax.numpy as jnp
import dataclasses

# from ml_collections import config_dict

@dataclasses.dataclass
class DataConfig:
    seed: int = 0
    dataset: str = "se"
    num_samples_in_epoch: int = int(2**14)


@dataclasses.dataclass
class SdeConfig:
    limiting_kernel: str = "white"


@dataclasses.dataclass
class OptimizationConfig:
    batch_size: int = 16
    num_epochs: int = 20
    ema_rate: float = 0.999


@dataclasses.dataclass
class NetworkConfig:
    num_bidim_attention_layers: int = 5
    hidden_dim: int = 16
    num_heads: int = 8


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
        num_samples_in_epoch=32
    ),
)


if __name__ == "__main__":
    from ml_collections import config_dict
    c = config_dict.ConfigDict(initial_dictionary=dataclasses.asdict(toy_config))

    from neural_diffusion_processes.ml_tools import config_utils

    config = config_utils.to_dataclass(Config, c)
    print(config_utils.get_id(config))
