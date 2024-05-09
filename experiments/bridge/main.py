import jax
import jax.numpy as jnp
import jax.scipy.special as sp
from functools import partial
import diffrax
import optax
from tqdm import tqdm
import sde.jax.markov_approximation as ma
from sde.jax.models import FractionalSDE, StaticFunction, DiagonalNormal
import flax.linen as nn
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import random
import seaborn as sns
sns.set(font_scale=1.5, rc={'text.usetex' : True}, style='whitegrid')
sns.set_context('paper')


OUTPUT_DIR = Path('experiments/bridge/output')


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1000)(x)
        x = nn.tanh(x)
        x = nn.Dense(1000)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


class ControlFunction:
    def __init__(self, num_k, num_latents):
        self.num_k = num_k
        self.num_latents = num_latents
        self.mlp = MLP()

    def init(self, key):
        params = self.mlp.init(key, jnp.zeros(self.num_latents * (self.num_k + 1) + 2))
        # Initialization trick from Glow.
        params['params']['Dense_2']['kernel'] *= 0
        return params

    def __call__(self, params, t, x, y, args):
        t = jnp.array([t])
        return self.mlp.apply(params, jnp.concatenate([jnp.sin(t), jnp.cos(t), x, y.flatten()], axis=-1))


def main(hurst, theta):
    seed = random.randint(0, 1000)
    random_key = jax.random.PRNGKey(seed)
    fbm_type = 1
    time_horizon = 6.
    num_k = 5
    gamma_max = 20.
    gamma = ma.gamma_by_gamma_max(num_k, gamma_max)
    sigma = .1
    time = 2.
    num_steps = 200
    dt = time / num_steps
    ts = jnp.linspace(0., time, num_steps + 1, endpoint=True)
    solver = diffrax.StratonovichMilstein()
    num_training_steps = 2000
    x0 = jnp.array([0.])

    b = StaticFunction(lambda t, x, args: - theta * x)
    u = ControlFunction(num_k, 1)
    s = StaticFunction(lambda *args: jnp.array([1.]))
    model = FractionalSDE(b, u, s, gamma, hurst=hurst, type=fbm_type, time_horizon=time_horizon)
    model.check_dt(dt)
    key, random_key = jax.random.split(random_key)
    params = model.init(key)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(params, key):
        xs, kl = model(params, key, x0, jnp.array([0., time]), dt, solver)
        neg_log_likelihood = (.5 * (xs[-1, 0]) ** 2 / sigma ** 2).sum()
        loss = neg_log_likelihood + kl
        return loss

    def batched_loss_fn(params, key, batch_size=32):
        keys = jax.random.split(key, batch_size)
        loss = jax.vmap(loss_fn, in_axes=(None, 0))(params, keys)
        return loss.mean()

    loss_grad = jax.jit(jax.value_and_grad(batched_loss_fn))
    loss_values = []
    for step in tqdm(range(num_training_steps)):
        random_key, key = jax.random.split(random_key)
        loss, grads = loss_grad(params, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss_values.append(loss)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f'hurst_{hurst:.2f}-theta_{theta:.2f}.pkl'
    with open(output_path, 'wb') as f:
        output = (hurst, params, loss_values)
        pickle.dump(output, f)

    if fbm_type == 1:
        if theta == 0.:
            def covariance(hurst, t, tau):
                return .5 * (jnp.abs(t) ** (2 * hurst) + jnp.abs(tau) ** (2 * hurst) - jnp.abs(t - tau) ** (2 * hurst))
        else:
            def integrate(f, t0, t1, num_steps=10000):
                t = jnp.linspace(t0, t1, num_steps + 2)
                t = t[1:-1]
                return jnp.where(t0 == t1, 0., jnp.trapz(jax.vmap(f)(t), t))
            @jax.jit
            def autocorrelation(hurst, t, theta, sigma=1.):
                # Lysy and Pillai
                t = jnp.abs(t)
                integral = integrate(lambda u: jnp.exp(theta * u) * u ** (2 * hurst - 2), 0., t)
                result = 1. ** 2 * hurst * (2 * hurst - 1) / 2 / theta * (sp.gamma(2 * hurst - 1) * (jnp.exp(- theta * t) + jnp.exp(theta * t) * sp.gammaincc(2 * hurst - 1, t)) / theta ** (2 * hurst - 1) + jnp.exp(- theta * t) * integral)
                return result
            def covariance(hurst, t, tau):
                return autocorrelation(hurst, t - tau, theta)
    else:
        raise NotImplementedError()
    def variance(t):
        k = partial(covariance, hurst)
        T = jnp.array([0., time])
        KtT = jax.vmap(k, (None, 0))(t, T)
        KTt = KtT.T
        KTT = jax.vmap(jax.vmap(k, (None, 0)), (0, None))(T, T)
        return k(t, t) - KtT @ jnp.linalg.inv(KTT + jnp.diag(jnp.array([1e-7, sigma ** 2]))) @ KTt

    # plot
    num_samples = 256

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key,  num_samples)
    xs, _ = jax.vmap(model, in_axes=(None, 0, None, None, None, None))(params, keys, x0, ts, dt, solver)
    mean_empirical = jnp.mean(xs, axis=0)
    std_empirical = jnp.std(xs, axis=0)

    ts_ = jnp.linspace(0., time, 1000 + 1, endpoint=True)
    variances = jax.vmap(variance)(ts_)
    std_analytical = jnp.sqrt(variances)

    fig, ax = plt.subplots(figsize=(3, 2))
    for xs_ in xs[:64]:
        ax.plot(ts, xs_, color='black', alpha=.05)

    ax.plot(ts_, - std_analytical, 'royalblue', linewidth=1.5, label='True Variance')
    ax.plot(ts_, std_analytical, 'royalblue', linewidth=1.5)

    ax.plot(ts, mean_empirical - std_empirical, '--', c='orange', linewidth=1.5, label='Empirical Variance')
    ax.plot(ts, mean_empirical + std_empirical, '--', c='orange', linewidth=1.5)

    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(-2, 2)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$X(t)$')
    # ax.legend(loc='upper right')
    fig.savefig(OUTPUT_DIR / f'bridge_hurst_{hurst:.2f}-theta_{theta:.2f}.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main(.1, 0.)
    main(.2, 0.)
    main(.3, 0.)
    main(.4, 0.)
    main(.5, 0.)
    main(.6, 0.)
    main(.7, 0.)
    main(.8, 0.)
    main(.9, 0.)
    main(.6, 1.)
    main(.7, 1.)
    main(.8, 1.)
    main(.9, 1.)
    main(.6, 2.)
    main(.7, 2.)
    main(.8, 2.)
    main(.9, 2.)
