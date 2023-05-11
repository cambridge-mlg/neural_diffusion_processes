"""
Script for training and evaluating a NDP on 1d regression data.

Modes
-----

1/ config.mode == train

Trains a new model from scratch and evaluates it on all the tasks
(interpolation, extrapolation and generalization).

2/ config.mode == smoketest

Trains a model without performing any "actions" during training,
at the end of the training loop the model is evaluated on a single batch of data
for all tasks.

3/ config.mode == eval and config.experiment_dir == ""

Build a model and evaluates it without training. This setting is typically
used in combination with `USE_TRUE_SCORE==True` (see Flags).

4/ config.mode == eval and config.experiment_dir == path_to_experiment

Restores a model using the configuration stored at experiment dir,
loads the latest weights for the model, and evaluates to model.

Flags
-----

1/ config.sde.exact_score == False

Default setting. Approximate the score by a neural network.

2/ config.sde.exact_score == True

For the GP datasets (squared exponential, matern and weakly periodic) we
use use the true score. In this case no training is required.
"""
from typing import Tuple, Mapping, Iterator, List, Optional, Callable
from jaxtyping import Float, Array

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import yaml
import numpy as np
import functools
import tqdm
import pathlib
import haiku as hk
import jaxkern
import gpjax
import jax
import jax.numpy as jnp
import jmp
import pandas as pd
import optax
import time
import datetime
import matplotlib.pyplot as plt
from absl import app
from pathlib import Path
from dataclasses import asdict

from ml_collections import config_dict, config_flags
from jax.config import config as jax_config

import neural_diffusion_processes as ndp
from neural_diffusion_processes.ml_tools.state import TrainingState, load_checkpoint, find_latest_checkpoint_step_index
from neural_diffusion_processes.ml_tools import config_utils
from neural_diffusion_processes import ml_tools
from neural_diffusion_processes.data import regression1d


try:
    from .config import Config, toy_config
except:
    from config import Config, toy_config


EXPERIMENT = "regression1d-May11"


_DATETIME = datetime.datetime.now().strftime("%b%d_%H%M%S")
_HERE = pathlib.Path(__file__).parent
_LOG_DIR = 'logs'


_CONFIG = config_flags.DEFINE_config_dict("config", config_utils.to_configdict(Config()))


def is_smoketest(config: Config) -> bool:
    return "smoketest" in config.mode


def get_experiment_name(config: Config):
    return f"{_DATETIME}_{config.data.dataset}_{str(config_utils.get_id(config))[:5]}"


def get_experiment_dir(config: Config, output: str = "root", exist_ok: bool = True) -> pathlib.Path:
    experiment_name = get_experiment_name(config)
    if is_smoketest(config):
        log_dir = f"{_LOG_DIR}"
        # log_dir = f"{_LOG_DIR}-smoketest"
    elif config.mode == "eval":
        log_dir = f"{_LOG_DIR}-eval"
    else:
        log_dir = f"{_LOG_DIR}"

    if output == "root":
        dir_ = _HERE / log_dir / experiment_name
    elif output == "plots":
        dir_ = _HERE / log_dir / experiment_name / "plots"
    elif output == "tensorboard":
        dir_ = _HERE / log_dir / "tensorboard" / experiment_name
    else:
        raise ValueError("Unknown output: %s" % output)

    dir_.mkdir(parents=True, exist_ok=exist_ok)
    return dir_


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


def get_log_prob(sde, network):
    """
    Returns a function which computes the log (conditional) probabability.

    Warning: The function can not be jitted!
    """

    @functools.partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    @jax.jit
    def delta_logp(params, x, y, mask, key):
        net = functools.partial(network, params)
        return ndp.sde.log_prob(sde, net, x, y, mask, key=key, rtol=None)

    def log_prob(params, x, y, mask, key):
        dlp, yT = delta_logp(params, x, y, mask, key)
        logp_prior = jax.vmap(sde.log_prob_prior)(
            x[:, ~mask[0].astype(jnp.bool_)], yT[:, ~mask[0].astype(jnp.bool_)]
        )
        return logp_prior + dlp
    
    def log_prob_cond(params, key,  x, y, mask, context):
        """
        params: []
        key: []
        x: [batch, num_pounts, x_dim]
        y: [batch, num_pounts, y_dim]
        mask: [batch, num_pounts]
        context: if not None:
            x: [batch, num_pounts_context, x_dim]
            y: [batch, num_pounts_context, y_dim]
            mask: [batch, num_pounts_context]
        """
        batch_size = len(x)
        if context is not None:
            assert len(context) == 3
            xc, yc, mc = context
            x = jnp.concatenate([x, xc], axis=1)
            y = jnp.concatenate([y, yc], axis=1)
            if mask is not None and mc is not None:
                mask = jnp.concatenate([mask, mc], axis=1)

            key, skey = jax.random.split(key)
            keys = jax.random.split(skey, batch_size)
            logp_context = log_prob(params, xc, yc, mc, keys)
        else:
            logp_context = 0.

        keys = jax.random.split(key, batch_size)
        logp_joint = log_prob(params, x, y, mask, keys)
        return logp_joint - logp_context
    
    return log_prob_cond


def m2i(mask):
    """mask to indices"""
    return ~mask.astype(jnp.bool_)


class Task:
    def __init__(self, key, task: str, dataset: str, batch_size: int, num_data: int, conditional_sampler: Callable, logp: Callable):
        self._key = key
        self._dataset = dataset
        self._task = task
        self._batch_size = batch_size
        self._num_data = num_data
        _key = jax.random.PRNGKey(0)
        # just for plotting - it doesn't matter that we use the same key across tasks/datasets.
        self._plt_batch = regression1d.get_batch(_key, batch_size, dataset, task)
        self._sampler = conditional_sampler
        self._logp = logp
    
    def plot2(self, params, key):
        """Plots continuous samples"""
        fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True, sharey=True, tight_layout=True)
        keys = jax.random.split(key, 10)
        lo = regression1d._TASK_CONFIGS[self._task].x_context_dist._low
        hi = regression1d._TASK_CONFIGS[self._task].x_context_dist._high
        xs = jnp.linspace(lo, hi, 50)[:, None]
        xs = jnp.tile(xs[None], [self._plt_batch.batch_size, 1, 1])
        samples = self._sampler(
            params,
            keys,
            self._plt_batch.xc,
            self._plt_batch.yc,
            None,
            xs,
            None,
        )
        for i, ax in enumerate(axes):
            # idx = m2i(self._plt_batch.mask_context[i])
            ax.plot(self._plt_batch.xc[i], self._plt_batch.yc[i], 'kx')
            ax.plot(self._plt_batch.xs[i], self._plt_batch.ys[i], 'rx')
            ax.plot(xs[i].ravel(), samples[:, i, :, 0].T, "C0", alpha=.2)

        return fig

    def plot(self, params, key):
        fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True, sharey=True, tight_layout=True)
        keys = jax.random.split(key, 10)
        samples = self._sampler(
            params,
            keys,
            self._plt_batch.xc,
            self._plt_batch.yc,
            None,
            self._plt_batch.xs,
            None,
        )
        means = jnp.mean(samples, axis=0)
        stds = jnp.std(samples, axis=0)

        fig.suptitle(self._task)
        for i, ax in enumerate(axes):
            ax.plot(self._plt_batch.xc[i], self._plt_batch.yc[i], 'kx')
            ax.plot(self._plt_batch.xs[i], self._plt_batch.ys[i], 'rx')
            ax.errorbar(
                jnp.reshape(self._plt_batch.xs[i], (-1,)),
                means[i].ravel(),
                yerr=stds[i].ravel() * 2,
                ecolor='C0',
                ls='none',
            )
            ax.plot(self._plt_batch.xs[i], means[i], 'C0.')

        return fig
    
    def eval(self, params, key):
        print(self._task)
        num_samples = 16
        # uses same key to keep test data fixed across evaluations
        # generator = data_generator(self._key, self._dataset, self._task, self._num_data, self._batch_size, num_epochs=1)
        ds = regression1d.get_dataset(self._dataset, self._task, key=self._key, batch_size=self._batch_size, samples_per_epoch=self._num_data, num_epochs=1)
        metrics = {
            # "mse": [],
            "loglik": []
        }
        pb = tqdm.tqdm(list(range(1, self._num_data // self._batch_size + 1)), miniters=1)
        for i, batch in zip(pb, ds):
            # key, *keys = jax.random.split(key, num_samples + 1)
            # samples = self._sampler(
            #     params,
            #     jnp.stack(keys),
            #     batch.xc,
            #     batch.yc,
            #     batch.mask_context,
            #     batch.xs,
            #     batch.mask,
            # )
            # y_pred = jnp.mean(samples, axis=0)  # [batch_size, num_points, y_dim]
            # mse = jnp.mean((batch.ys - y_pred) ** 2, axis=[1, 2])  # [batch_size]
            # metrics["mse"].append(mse)

            key, skey = jax.random.split(key)
            context = (batch.xc, batch.yc, batch.mask_context)
            logp_cond = self._logp(
                params, skey, batch.xs, batch.ys, batch.mask, context
            )
            logp_cond = logp_cond / batch.num_targets
            metrics["loglik"].append(jnp.reshape(logp_cond, (-1)))

            v = jnp.mean(jnp.stack(metrics["loglik"]))
            pb.set_description(f"loglik {v:.4f}")

        
        err = lambda v: 1.96 * jnp.std(v) / jnp.sqrt(len(v))
        summary_stats = [
            ("mean", jnp.mean),
            ("std", jnp.std),
            ("err", err)
        ]
        metrics = {f"{k}_{n}": s(jnp.stack(v)) for k, v in metrics.items() for n, s in summary_stats}
        print(metrics)
        return metrics

    def get_callback(self, writer: ml_tools.writers._MetricWriter):

        def callback(step, t, **kwargs):
            del t
            params = kwargs["state"].params_ema
            key = kwargs["key"]
            fig = self.plot(params, key)
            writer.write_figures(step, {f"{self._task}": fig})
            if "extrapolation" not in self._task:
                fig2 = self.plot2(params, key)
                writer.write_figures(step, {f"{self._task}_cont": fig2})

            metrics = self.eval(params, key)
            writer.write_scalars(step, {f"{self._task}_{k}": v for k, v in metrics.items()})
            
        def save_callback(step, t, **kwargs):
            try:
                callback(step, t, **kwargs)
            except Exception as e:
                print(">>>>>> Error during callback <<<<<")
                print(e)

        return save_callback


def main(_):
    jax_config.update("jax_enable_x64", True)
    policy = jmp.get_policy('params=float32,compute=float32,output=float32')

    config = config_utils.to_dataclass(Config, _CONFIG.value)

    experiment_dir_if_exists = Path(config.experiment_dir)
    if config.mode == "eval" and (experiment_dir_if_exists / "config.yaml").exists():
        print("****** eval mode:")
        print("Restoring old configuration")
        config_path = experiment_dir_if_exists / "config.yaml"
        restored_config = yaml.safe_load(config_path.read_text())
        restored_config = config_utils.to_dataclass(Config, restored_config)
        # overwrite values by restored config such that the model can be loaded,
        # weights restored.
        config.data = restored_config.data
        config.sde = restored_config.sde
        config.network = restored_config.network
    elif config.mode == "eval":
        print("****** eval mode:")
        f = str((experiment_dir_if_exists / "config.yaml").absolute())
        print(f"Building and evaluating new model as {f} did not exist.")


    key = jax.random.PRNGKey(config.seed)
    key_iter = _get_key_iter(key)

    ####### init relevant diffusion classes
    beta = ndp.sde.LinearBetaSchedule(t0=config.sde.t0)
    if "short" in config.sde.limiting_kernel:
        short_lengthscale = True
        limiting_kernel = config.sde.limiting_kernel[config.sde.limiting_kernel.find("-")+1:]
    else:
        short_lengthscale = False
        limiting_kernel = config.sde.limiting_kernel

    limiting_kernel = limiting_kernel[config.sde.limiting_kernel.find("-")+1:]
    limiting_kernel = ndp.kernels.get_kernel(limiting_kernel, active_dims=[0])
    hyps = {
        "mean_function": {},
        "kernel": limiting_kernel.init_params(None),
    }

    if short_lengthscale and "lengthscale" in hyps["kernel"]:
        hyps["kernel"]["lengthscale"] = config.sde.limiting_kernel_lengthscale * hyps["kernel"]["lengthscale"]

    if "noisy" in config.sde.limiting_kernel:
        limiting_kernel = jaxkern.SumKernel(
            [limiting_kernel, jaxkern.stationary.White(active_dims=[0])]
        )
        v = config.sde.limiting_kernel_noise_variance
        hyps["kernel"]["variance"] = 1. - v
        hyps["kernel"] = [hyps["kernel"], {"variance": config.sde.limiting_kernel_noise_variance}]

    sde = ndp.sde.SDE(
        limiting_kernel,
        gpjax.mean_functions.Zero(),
        hyps,
        beta,
        score_parameterization=ndp.sde.ScoreParameterization.get(
            config.sde.score_parametrization
        ),
        # Below parameterisations are all set to False if we use the true score.
        std_trick=config.sde.std_trick,
        residual_trick=config.sde.residual_trick,
        loss_type=config.sde.loss,
        exact_score=config.sde.exact_score,
    )

    if config.sde.exact_score:
        factory = regression1d._DATASET_FACTORIES[config.data.dataset] 
        assert isinstance(factory, regression1d.GPFunctionalDistribution)
        mean0, kernel0, params0 = factory.mean, factory.kernel, factory.params
        true_score_network = sde.get_exact_score(mean0, kernel0, params0)
    else:
        true_score_network = None

    ##### Plot a training databatch
    batch0 = regression1d.get_batch(next(key_iter), 2, config.data.dataset, "training")
    _, ax = plt.subplots()
    ax.plot(batch0.xc[..., 0].T, batch0.yc[..., 0].T, "C0.", label="context")
    ax.plot(batch0.xs[..., 0].T, batch0.ys[..., 0].T, "C1.", label="target")
    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc='best')
    plt.savefig(str(get_experiment_dir(config, "plots") / "data.png"))


    ####### Forward haiku model
    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x, mask):
        t, y, x = policy.cast_to_compute((t, y, x))
        model = ndp.models.attention.BiDimensionalAttentionModel(
            n_layers=config.network.num_bidim_attention_layers,
            hidden_dim=config.network.hidden_dim,
            num_heads=config.network.num_heads,
            translation_invariant=config.network.translation_invariant,
        )
        return model(x, y, t, mask)

    @jax.jit
    def net(params, t, yt, x, mask, *, key):
        del key  # the network is deterministic
        #NOTE: Network awkwardly requires a batch dimension for the inputs
        return network.apply(params, t[None], yt[None], x[None], mask[None])[0]

    def loss_fn(params, batch: ndp.data.DataBatch, key):
        net_params = functools.partial(net, params)
        return ndp.sde.loss(sde, net_params, batch, key)

    num_steps_per_epoch = config.data.num_samples_in_epoch // config.optimization.batch_size
    num_steps = num_steps_per_epoch * config.optimization.num_epochs
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.optimization.init_lr,
        peak_value=config.optimization.peak_lr,
        warmup_steps=num_steps_per_epoch * config.optimization.num_warmup_epochs,
        decay_steps=num_steps_per_epoch * config.optimization.num_decay_epochs,
        end_value=config.optimization.end_lr,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )
    
    @jax.jit
    def init(batch: ndp.data.DataBatch, key) -> TrainingState:
        key, init_rng = jax.random.split(key)
        t = 1. * jnp.zeros((batch.ys.shape[0]))
        initial_params = network.init(init_rng, t=t, y=batch.ys, x=batch.xs, mask=batch.mask)
        initial_params = policy.cast_to_param((initial_params))
        initial_opt_state = optimizer.init(initial_params)
        return TrainingState(
            params=initial_params,
            params_ema=initial_params,
            opt_state=initial_opt_state,
            key=key,
            step=0,
        )

    @jax.jit
    def update_step(state: TrainingState, batch: ndp.data.DataBatch) -> Tuple[TrainingState, Mapping]:
        new_key, loss_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn)
        loss_value, grads = loss_and_grad_fn(state.params, batch, loss_key)
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * config.optimization.ema_rate
            + p * (1.0 - config.optimization.ema_rate),
            state.params_ema,
            new_params,
        )
        new_state = TrainingState(
            params=new_params,
            params_ema=new_params_ema,
            opt_state=new_opt_state,
            key=new_key,
            step=state.step + 1
        )
        metrics = {
            'loss': loss_value,
            'step': state.step
        }
        return new_state, metrics

    state = init(batch0, jax.random.PRNGKey(config.seed))
    
    if config.mode == "eval" and (experiment_dir_if_exists / "checkpoints").exists():
        index = find_latest_checkpoint_step_index(str(experiment_dir_if_exists))
        state = load_checkpoint(state, str(experiment_dir_if_exists), step_index=index)
        print("Successfully loaded checkpoint @ step {}".format(state.step))

    
    ########## Plotting

    @functools.partial(jax.vmap, in_axes=[None, 0, None, None, None, None, None])
    @functools.partial(jax.vmap, in_axes=[None, None, 0, 0, 0, 0, 0])
    @jax.jit
    def conditional(params, key, xc, yc, maskc, xs, mask):
        net_params = (
            true_score_network if config.sde.exact_score else functools.partial(net, params)
        )
        return ndp.sde.conditional_sample2(sde, net_params, xc, yc, xs, key=key, mask_context=maskc, mask_test=mask)

    logp = get_log_prob(
        sde,
        lambda params, *args, **kwargs: true_score_network(*args, **kwargs) if config.sde.exact_score else \
            net(params, *args, **kwargs)
    )
    
    experiment = EXPERIMENT + ("-smoketest" if is_smoketest(config) else "")
    exp_root_dir = get_experiment_dir(config)
    local_writer = ml_tools.writers.LocalWriter(str(exp_root_dir), flush_every_n=100)
    tb_writer = ml_tools.writers.TensorBoardWriter(get_experiment_dir(config, "tensorboard"))
    aim_writer = ml_tools.writers.AimWriter(experiment)
    writer = ml_tools.writers.MultiWriter([aim_writer, tb_writer, local_writer])

    hparams = {"experiment": experiment, **asdict(config)}
    writer.log_hparams(hparams)

    if is_smoketest(config):
        tasks = ["interpolation"]
    else:
        tasks = ["interpolation"] #, "generalization"]  # extrapolation

    tasks = [
        Task(
            next(key_iter),
            task,
            config.data.dataset,
            batch_size=config.eval.batch_size,
            num_data=config.eval.batch_size,  # run for single batch (quick)
            conditional_sampler=conditional,
            logp=logp
        )
        for task in tasks
    ]

    task_callbacks = [
        task.get_callback(writer) for task in tasks
    ]

    
    @functools.partial(jax.vmap, in_axes=[None, None, 0])
    def prior(params, x_target, key):
        net_params = (
            true_score_network if config.sde.exact_score else functools.partial(net, params)
        )
        return ndp.sde.sde_solve(sde, net_params, x_target, key=key)
    
    def callback_plot_prior(state: TrainingState, key):
        params = state.params_ema
        xx = jnp.linspace(-2, 2, 60)[:, None]
        samples = prior(params, xx, jax.random.split(key, 16))
        fig, ax = plt.subplots()
        ax.plot(xx, samples[:, -1, :, 0].T, "C0", alpha=.3)
        return {"prior": fig}


    train_ds = regression1d.get_dataset(
        config.data.dataset,
        "training",
        key=next(key_iter),
        samples_per_epoch=config.data.num_samples_in_epoch,
        batch_size=config.optimization.batch_size,
        num_epochs=config.optimization.num_epochs,
    )

    if config.mode == "eval": num_steps = 0

    actions = [
        ml_tools.actions.PeriodicCallback(
            every_steps=10,
            callback_fn=lambda step, t, **kwargs: writer.write_scalars(step, kwargs["metrics"])
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=None if is_smoketest(config) else num_steps // 4,
            callback_fn=lambda step, t, **kwargs: [
                cb(step, t, **kwargs) for cb in task_callbacks
            ]
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=num_steps//10,
            callback_fn=lambda step, t, **kwargs: ml_tools.state.save_checkpoint(kwargs["state"], exp_root_dir, step)
        )
    ]

    progress_bar = tqdm.tqdm(list(range(1, num_steps + 1)), miniters=1)

    for step, batch, key in zip(progress_bar, train_ds, key_iter):
        if config.sde.exact_score:
            metrics = {'loss': 0.0, 'step': step}
        else:
            state, metrics = update_step(state, batch)
        metrics["lr"] = learning_rate_schedule(step)

        for action in actions:
            action(step, t=None, metrics=metrics, state=state, key=key)

        if step % 100 == 0:
            progress_bar.set_description(f"loss {metrics['loss']:.2f}")


    for task, callback in zip(tasks, task_callbacks):
        if not is_smoketest(config):
            task._num_data = config.eval.num_samples_in_epoch
        callback(num_steps + 1, None, state=state, key=next(key_iter))
    
    

if __name__ == "__main__":
    app.run(main)
