import os
import sys

import jax
import jax.numpy as jnp
from einops import rearrange

'''
We write a class which get an input X,Y and translates it normalized values
By normalized, we mean here:
    - the mean of X is 0 and the radius 10, there are two types to convert X, first
    from the US grid to [-10,10] or from the China grid to [-10,10]
    - the values of Y are also transformed (scalar values are centered and rescaled while vectors 
    are only rescaled)
'''
class ERA5_translater(object):
    def __init__(self, place='US'):
        self.place=place
        if self.place=='US':
            self.X_mean=jnp.array([-91.,35.], dtype=float)
        elif self.place=='China':
            self.X_mean=jnp.array([110.,30.], dtype=float)
        else:
            sys.exit("Unknown place.")
        
        self.Y_mean=jnp.array([100.1209,   7.4628,   0.0000,   0.0000], dtype=float)
        self.Y_std=jnp.array([1.4738, 8.5286, 3.4162, 3.4162], dtype=float)
        self.Y_mean_out=jnp.array([0.0000,   0.0000], dtype=float)
        self.Y_std_out=jnp.array([3.4162, 3.4162], dtype=float)
        
    def norm_X(self, X):
        '''
        X - torch.Tensor - shape (*,2)
        --> returns torch.Tensor with same shape translated and scaled with mean and std for self.X_Tensor
        '''
        # return(2*X.sub(self.X_mean[None,:]))
        return (2*(X - self.X_mean[None,:]))

    def denorm_X(self, X):
        '''
        X - torch.Tensor - shape (*,2)
        --> Inverse of self.norm_X
        '''
        # return(X.div(2).add(self.X_mean[None,:]))
        return ((X / 2) + self.X_mean[None,:])
    
    def norm_Y(self, Y):
        '''
        Y - torch.Tensor - shape (*,4)
        --> returns torch.Tensor with same shape translated and scaled with mean and std for self.Y_data
        '''
        # return(Y.sub(self.Y_mean[None,:]).div(self.Y_std[None,:]))
        return ((Y - self.Y_mean[None,:]) / self.Y_std[None,:])

    def denorm_Y(self, Y):
        '''
        Y - torch.Tensor - shape (*,4)
        --> Inverse of self.norm_Y
        '''
        # return(Y.mul(self.Y_std[None,:]).add(self.Y_mean[None,:]))
        return (Y * self.Y_std[None,:] + self.Y_mean[None,:])
    
    def denorm_Y_out(self, Y):
        '''
        Y - torch.Tensor - shape (*,4)
        --> Inverse of self.norm_Y
        '''
        # return(Y.mul(self.Y_std_out[None,:]).add(self.Y_mean_out[None,:]))
        return (Y * self.Y_std_out[None,:] + self.Y_mean_out[None,:])
    
    def translate_to_normalized_scale(self, X, Y):
        '''
        X - torch.Tensor - shape (*,2)
        Y - torch.Tensor - shape (*,4)
        --> returns torch.Tensor with same shape translated and scaled with mean and std
        '''
        return (self.norm_X(X), self.norm_Y(Y))
    
    def translate_to_original_scale(self, X, Y):
        '''
        X - torch.Tensor - shape (*,2)
        Y - torch.Tensor - shape (*,4)
        --> returns torch.Tensor with same shape translated and scaled with mean and std
        '''
        return (self.denorm_X(X), self.denorm_Y(Y))


class ERA5Dataloader:
    def __init__(self, batch_size, data_dir, file_ending, rng=jax.random.PRNGKey(0)):
        self.batch_size = batch_size

        self.data = rearrange(
            jnp.load(os.path.join(data_dir, f"data_{file_ending}.npy")),
            "n h w d -> n (h w) d",
        )
        self.time = jnp.load(
            os.path.join(data_dir, f"time_{file_ending}.npy"), allow_pickle=True
        )
        self.latlon = rearrange(
            jnp.load(os.path.join(data_dir, f"latlon_{file_ending}.npy")),
            "h w d -> (h w) d",
        )

        print("data", self.data.shape)
        self.rng = rng

    def __len__(self):
        return self.data.shape[0]

    def __next__(self):
        self.rng, next_rng = jax.random.split(self.rng)
        indices = jax.random.choice(next_rng, len(self), shape=(self.batch_size,))

        return self.data[indices], jnp.repeat(
            self.latlon[None, ...], axis=0, repeats=self.batch_size
        )


def ERA5Dataset(key, data_dir, file_ending, dataset="train", place="US", **kwargs):
    """"
    ['sp_in_kPa','t_in_Cels','wind_10m_east','wind_10m_north']
     """
    data = rearrange(
        jnp.load(os.path.join(data_dir, f"data_{file_ending}.npy")),
        "n h w d -> n (h w) d",
    )
    # time = jnp.load(
    #     os.path.join(data_dir, f"time_{file_ending}.npy"), allow_pickle=True
    # )

    latlon = rearrange(
        jnp.load(os.path.join(data_dir, f"latlon_{file_ending}.npy")),
        "h w d -> (h w) d",
    )

    # print(data.mean((0, 1)))
    # print(data.std((0, 1)))
    y = data[..., [-2, -1, 2, 3]] # pressure, temperature, 10m wind east, 10m wind north
    # print(y.mean((0, 1)))
    # print(y.std((0, 1)))
    # y = y - jnp.array([0., 273.15, 0., 0.])
    # y = y / jnp.array([1000., 1., 1., 1.])
    # print(y.mean((0, 1)))
    # print(y.std((0, 1)))

    x = jnp.repeat(latlon[None, ...], data.shape[0], 0)
    x = x[..., [1, 0]]

    # normalize!
    y = jax.nn.normalize(y, (0, 1))
    x = jax.nn.normalize(x, (0, 1))
    # translater = ERA5_translater(place=place)
    # x, y = translater.translate_to_normalized_scale(x, y)

    if dataset == "train":
        x = x[:3500]
        y = y[:3500]
    else:
        x = x[3500:]
        y = y[3500:]

    x = jnp.array(x, dtype=float)
    y = jnp.array(y, dtype=float)

    print("x", x.shape)
    print("y", y.shape)
    return x, y
