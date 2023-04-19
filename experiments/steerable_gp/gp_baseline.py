import math
from typing import Tuple, Mapping, Iterator, List
import os
import socket
import logging
import yaml
from collections import defaultdict
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate, call
from jax.config import config as jax_config

import neural_diffusion_processes as ndp
from neural_diffusion_processes.utils.loggers_pl import LoggerCollection
from neural_diffusion_processes.utils import flatten, unflatten


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


log = logging.getLogger(__name__)

def run(cfg):
    jax.config.update("jax_enable_x64", True)

    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, cfg.paths.ckpt_dir)
    wandb_cfg_path = os.path.join(run_path, "wandb", "config.yaml")
    os.makedirs(os.path.dirname(wandb_cfg_path), exist_ok=True)

    os.makedirs(ckpt_path, exist_ok=True)
    loggers = [instantiate(logger_config) for logger_config in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    key = jax.random.PRNGKey(cfg.seed)
    key_iter = _get_key_iter(key)

 ####### prepare data
    data_test = call(
        cfg.data,
        key=jax.random.PRNGKey(cfg.data.seed_test),
        num_samples=cfg.data.num_samples_test,
        dataset="test"
    )
    dataloader_test = ndp.data.dataloader(
        data_test,
        batch_size=cfg.eval.batch_size,
        # batch_size=100,
        key=next(key_iter),
        run_forever=False,  # only run once
        n_points=cfg.data.n_points,
    )
    data_test: List[ndp.data.DataBatch] = [
        ndp.data.split_dataset_in_context_and_target(batch, next(key_iter), cfg.data.min_context, cfg.data.max_context)
        for batch in dataloader_test
    ]

    cfg.data._target_ = "neural_diffusion_processes.data.get_vec_gp_cond"
    true_posterior = call(cfg.data)
    metrics = defaultdict(list)

    n_passes = 5
    for n in range(n_passes):
        for i, batch in enumerate(data_test):
            print(i, batch.xs.shape, batch.ys.shape)
            
            # predictive log-likelihood
            n_test = batch.ys.shape[-2]
            true_cond_logp = jax.vmap(lambda xc, yc, x, y: true_posterior(xc, yc, x).log_prob(flatten(y)))(batch.xc, batch.yc, batch.xs, batch.ys)
            metrics["true_cond_logp"].append(jnp.mean(true_cond_logp / n_test))

    # NOTE: currently assuming same batch size, should use sum and / len(data_test) instead?
    v = {k: jnp.mean(jnp.stack(v)) for k, v in metrics.items()}
    logger.log_metrics(v, cfg.optim.num_steps)
    logger.save()
    print(v)


@hydra.main(config_path="config", config_name="main", version_base="1.3.2")
def main(cfg):
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["WANDB_START_METHOD"] = "thread"
    return run(cfg)


if __name__ == "__main__":
    main()