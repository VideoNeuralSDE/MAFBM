import jax
import jax.numpy as jnp
import numpy as onp
import flax.linen as nn
import matplotlib.pyplot as plt
import optax
import diffrax
import distrax
import sde.jax.markov_approximation as ma
from sde.jax.models import FractionalSDE, VideoSDE, StaticFunction
from sde import data
from sde.jax.util import NumpyLoader
from moviepy.editor import ImageSequenceClip
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import typing
from jsonargparse import ArgumentParser, ActionConfigFile
import wandb


class MLP(nn.Module):
    num_outputs: int
    activation: typing.Callable = lambda x: x

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(200)(x)
        x = nn.tanh(x)
        x = nn.Dense(200)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.num_outputs)(x)
        x = self.activation(x)
        return x


class ControlFunction:
    def __init__(self, num_k, num_latents, num_features):
        self.num_k = num_k
        self.num_latents = num_latents
        self.num_features = num_features
        self.mlp = MLP(num_latents)

    def init(self, key):
        params = self.mlp.init(key, jnp.zeros(self.num_latents * (self.num_k + 1) + self.num_features))
        # Initialization trick from Glow.
        params['params']['Dense_2']['kernel'] *= 0
        return params

    def __call__(self, params, t, x, y, args):
        context = args['context']
        h = jax.vmap(jnp.interp, (None, None, 1))(t, context['ts'], context['hs'])
        return self.mlp.apply(params, jnp.concatenate([x, y.flatten(), h], axis=-1))


class Drift:
    def __init__(self, num_latents):
        self.num_latents = num_latents
        self.mlp = MLP(num_latents)

    def init(self, key):
        params = self.mlp.init(key, jnp.zeros(self.num_latents))
        return params

    def __call__(self, params, t, x, args):
        return self.mlp.apply(params, x)


class Diffusion:
    # commutative noise!
    def __init__(self, num_latents):
        self.num_latents = num_latents
        self.mlp = MLP(1, nn.softplus)

    def init(self, key):
        keys = jax.random.split(key, self.num_latents)
        params = jax.vmap(self.mlp.init)(keys, jnp.zeros((self.num_latents, 1)))
        return params

    def __call__(self, params, t, x, args):
        return jax.vmap(self.mlp.apply)(params, x[:, None])[:, 0]


def build_data_and_model(
        dataset: str,
        white: bool,
        num_latents: int,
        num_contents: int,
        num_features: int,
        num_k: int,
        gamma_max: float,
        int_sub_steps: int,
    ):

    if white:
        num_k = 1
        gamma = None
        hurst = - 1
    else:
        gamma = ma.gamma_by_gamma_max(num_k, gamma_max)
        hurst = None

    data_train, data_val, dataset_kwargs = data.get(dataset)
    ts = jnp.arange(len(data_train[0])) * dataset_kwargs['dt']
    dt = dataset_kwargs['dt'] / int_sub_steps

    key = jax.random.PRNGKey(0)
    b = Drift(num_latents)
    u = ControlFunction(num_k, num_latents, num_features)
    s = Diffusion(num_latents)
    sde = FractionalSDE(b, u, s, gamma, hurst=hurst, type=1, time_horizon=ts[-1], num_latents=num_latents)
    x0_prior = distrax.MultivariateNormalDiag(jnp.zeros(num_latents), jnp.ones(num_latents))
    model = VideoSDE(dataset_kwargs['image_size'], dataset_kwargs['num_channels'], num_features, num_latents, num_contents, x0_prior, True, sde)
    model._sde.check_dt(dt)
    params = model.init(key)
    return ts, dt, data_train, data_val, model, params


def train(
        dataset: str,
        white: bool = False,    # fallback to standard sde
        batch_size: int = 32,
        num_epochs: int = 100,
        num_latents: int = 4,
        num_contents: int = 64,
        num_features: int = 64,
        num_k: int = 5,
        gamma_max: float = 20.,
        int_sub_steps: int = 3,
        kl_weight: float = 1.,
    ):
    solver = diffrax.StratonovichMilstein()

    ts, dt, data_train, data_val, model, params = build_data_and_model(dataset, white, num_latents, num_contents, num_features, num_k, gamma_max, int_sub_steps)
    dataloader = NumpyLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    def loss_fn(params, key, frames):
        frames_, (kl_x0, logpath) = model(params, key, ts, frames, dt, solver)
        nll = ((frames - frames_) ** 2).sum()
        loss = nll + kl_weight * (kl_x0 + logpath)
        return loss, (nll, kl_x0, logpath)

    def batched_loss_fn(params, key, frames, batch_size=batch_size):
        keys = jax.random.split(key, batch_size)
        loss, aux = jax.vmap(loss_fn, (None, 0, 0))(params, keys, frames)
        return loss.mean(), jax.tree_util.tree_map(jnp.mean, aux)

    loss_grad = jax.jit(jax.value_and_grad(batched_loss_fn, has_aux=True))

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    random_key = jax.random.PRNGKey(7)
    for epoch in range(num_epochs):
        pbar = tqdm(range(len(dataloader)))
        for step, frames in zip(pbar, dataloader):
            random_key, key = jax.random.split(random_key)
            (loss, loss_aux), grads = loss_grad(params, key, frames)
            nll, kl_x0, logpath = loss_aux
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            pbar.set_description(f'[Epoch {epoch+1}/{num_epochs}] Loss: {float(loss):.2f}, Hurst: {model._sde.hurst(params["sde"]):.2f}, NLL: {nll:.2f}, KL_x0: {kl_x0:.2f}, KL_path: {logpath:.2f}')

            if onp.isnan(float(loss)):
                return

            wandb.log({
                'loss': float(loss),
                'nll': float(nll),
                'kl_x0': float(kl_x0),
                'kl_path': float(logpath),
                'hurst': float(model._sde.hurst(params["sde"])),
            })

        with open('params.p', 'wb') as f:
            pickle.dump(params, f)
        wandb.save('params.p')

    wandb.join(quiet=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_function_arguments(train, as_positional=False)

    cfg = parser.parse_args()
    wandb.init(project=f'jax-{cfg.dataset}', config=cfg)
    train(**cfg)
