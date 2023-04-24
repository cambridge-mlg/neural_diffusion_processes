# from pathlib import Path
# import sys
# import click

# from neural_diffusion_processes.ml_tools.state import find_latest_checkpoint_step_index, load_checkpoint, TrainingState


# @click.command()
# @click.argument('path')
# def main(path):
#     """Simple program that greets NAME for a total of COUNT times."""

#     @hk.without_apply_rng
#     @hk.transform
#     def network(t, y, x, mask):
#         t, y, x = policy.cast_to_compute((t, y, x))
#         model = ndp.models.attention.BiDimensionalAttentionModel(
#             n_layers=config.network.num_bidim_attention_layers,
#             hidden_dim=config.network.hidden_dim,
#             num_heads=config.network.num_heads,
#             translation_invariant=config.network.translation_invariant,
#         )
#         return model(x, y, t, mask)

#     @jax.jit
#     def net(params, t, yt, x, mask, *, key):
#         del key  # the network is deterministic
#         #NOTE: Network awkwardly requires a batch dimension for the inputs
#         return network.apply(params, t[None], yt[None], x[None], mask[None])[0]



#     exp_dir = Path(path)
#     index = find_latest_checkpoint_step_index(str(exp_dir))

#     ckpt = load_checkpoint(
#         TrainingState(None, None, None, None, None),
#         str(exp_dir),
#         step_index=index
#     )
#     print(ckpt)



# if __name__ == '__main__':
#     main()