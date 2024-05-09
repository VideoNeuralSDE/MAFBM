import jax
import jax.numpy as jnp
import equinox as eqx
import sde.jax.markov_approximation as ma
import jax.scipy.special as sp
import flax.linen as nn
import distrax
import diffrax


class Beta:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    @property
    def mu(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def nu(self):
        return self.alpha + self.beta

    def sample(self, key):
        return jax.random.beta(key, self.alpha, self.beta)

    def kl_divergence(self, other):
        return sp.betaln(other.alpha, other.beta) - sp.betaln(self.alpha, self.beta) + (self.alpha - other.alpha) * sp.digamma(self.alpha) + (self.beta - other.beta) * sp.digamma(self.beta) + (other.alpha - self.alpha + other.beta - self.beta) * sp.digamma(self.alpha + self.beta)


def up(x):
    shape = x.shape
    new_shape = [*shape[:-3], 2 * shape[-3], 2 * shape[-2], shape[-1]]
    return jax.image.resize(x, new_shape, 'nearest')


class DownBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.GroupNorm(8)(x)
        x = nn.silu(x)
        return x


class UpBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.GroupNorm(8)(x)
        x = up(x)
        x = nn.silu(x)
        return x


class Encoder(nn.Module):
    image_size: int
    num_channels: int
    num_features: int
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        assert self.image_size == 64
        x = DownBlock(self.num_features)(x)
        x = DownBlock(2 * self.num_features)(x)
        x = DownBlock(4 * self.num_features)(x)
        x = DownBlock(4 * self.num_features)(x)
        x_flat = x.reshape(x.shape[:-3] + (-1,))
        x = nn.Dense(self.num_outputs)(x_flat)
        return x


class Decoder(nn.Module):
    image_size: int
    num_channels: int
    num_features: int
    num_latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * 4 * 4 * self.num_features)(x)
        x = x.reshape(x.shape[:-1] + (4, 4, 4 * self.num_features))
        x = UpBlock(4 * self.num_features)(x)
        x = UpBlock(2 * self.num_features)(x)
        x = UpBlock(self.num_features)(x)
        x = UpBlock(self.num_features)(x)
        x = nn.Conv(self.num_features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.silu(x)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.sigmoid(x)
        return x


class Content(nn.Module):
    num_features: int
    num_contents: int
    num_content_frames: int

    @nn.compact
    def __call__(self, h):
        w = jnp.median(h[:self.num_content_frames], axis=-2)
        w = nn.Dense(self.num_features)(w)
        w = nn.silu(w)
        w = nn.Dense(self.num_contents)(w)
        return w


class Infer(nn.Module):
    num_features: int
    num_latents: int

    @nn.compact
    def __call__(self, x):
        num_frames, num_features = x.shape
        h = nn.Conv(self.num_features, kernel_size=(3,), padding='SAME')(x)
        h = nn.silu(h)
        h = nn.Conv(self.num_features, kernel_size=(3,), padding='SAME')(h)

        g = nn.Dense(self.num_features)(jnp.concatenate([h[0], x[0], x[1], x[2]], axis=-1))
        g = nn.silu(g)
        g = nn.Dense(self.num_features)(g)
        g = nn.silu(g)
        g = nn.Dense(2 * self.num_latents)(g)

        x0_mean = g[:self.num_latents]
        x0_logvar = g[self.num_latents:]
        x0_posterior = distrax.MultivariateNormalDiag(x0_mean, jnp.exp(.5 * x0_logvar))
        return x0_posterior, h


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1000)(x)
        x = nn.tanh(x)
        x = nn.Dense(1000)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        x = nn.tanh(x)
        return x


class Function:
    def init(self, key):
        return {}

    def __call__(self, params, *args):
        raise NotImplementedError


class StaticFunction(Function):
    def __init__(self, function):
        self.function = function

    def init(self, key):
        return {}

    def __call__(self, params, *args):
        return self.function(*args)


class FractionalSDE:
    """
    Neural Stochastic Differential Equations driven by fractional Brownian Motion.
    Args:
        b (Function): The drift function of the SDE.
        u (Function): The control function of the SDE.
        s (Function): The diffusion function of the SDE.
        gamma (jnp.ndarray): The gamma values of the Ornstein-Uhlenbeck processes used to approximate fractional Brownian Motion.
        hurst (float or None): The Hurst coefficient of the fractional Brownian Motion. If None, then the Hurst coefficient is learnable. If -1, the model falls back to standard Brownian Motion (gamma = [0], omega = [1]).
        type [1, 2] (int): The type of the fractional Brownian Motion. 1 for type I, 2 for type II.
        time_horizon (float): The time horizon of the model, used to calculate omega.
        num_latents (int): The number of latent dimensions.
    """
    def __init__(
            self,
            b: Function,
            u: Function,
            s: Function,
            gamma: jnp.ndarray,
            hurst: float or None,
            type: int = 1,
            time_horizon: float = 1.,
            num_latents: int = 1,
        ):
        self.gamma = gamma
        self.type = type
        self.num_latents = num_latents
        self._b = b
        self._u = u
        self._s = s

        if type == 1:
            self.omega_fn = jax.jit(lambda hurst: ma.omega_optimized_1(self.gamma, hurst, time_horizon))
        elif type == 2:
            self.omega_fn = jax.jit(lambda hurst: ma.omega_optimized_2(self.gamma, hurst, time_horizon))
        else:
            raise ValueError('type must be either 1 or 2')

        if hurst is None:
            self._hurst = None
        elif hurst < 0:
            print('Falling back to standard Brownian Motion (gamma = [0], omega = [1]). Args gamma and type are ignored.')
            self._hurst = .5
            self.type = 2   # prevent problems with gamma = 0
            self.gamma = jnp.array([0.])
            self._omega = jnp.array([1.])
        else:
            self._hurst = hurst
            self._omega = self.omega_fn(hurst)

    @property
    def num_k(self):
        return len(self.gamma)

    def check_dt(self, dt):
        assert self.gamma.max() * dt < .5, 'dt too large for stable integration, please reduce dt or decrease largest gamma'

    def init(self, key):
        keys = jax.random.split(key, 3)
        params = {}

        if self._hurst is None:
            params['hurst_raw'] = 0.    # sigmoid(0.) = .5

        params['b'] = self._b.init(keys[0])
        params['u'] = self._u.init(keys[1])
        params['s'] = self._s.init(keys[2])
        return params

    def hurst(self, params):
        if self._hurst is None:
            return jax.nn.sigmoid(params['hurst_raw'])
        else:
            return self._hurst

    def omega(self, hurst):
        if self._hurst is None:
            return self.omega_fn(hurst)
        else:
            return self._omega

    def b(self, params, t, x, args):      # Prior drift.
        return self._b(params['b'], t, x, args)

    def u(self, params, t, x, y, args):   # Approximate posterior control.
        return self._u(params['u'], t, x, y, args)

    def s(self, params, t, x, args):      # Shared diffusion.
        return self._s(params['s'], t, x, args)

    def __call__(self, params, key, x0, ts, dt, solver='euler', args=None):
        keys = jax.random.split(key, 4)

        hurst = self.hurst(params)
        omega = self.omega(hurst)

        if self.type == 1:
            cov = 1 / (self.gamma[None, :] + self.gamma[:, None])
            y0 = jax.random.multivariate_normal(keys[2], jnp.zeros((self.num_latents, self.num_k)), cov)
        elif self.type == 2:
            y0 = jnp.zeros((self.num_latents, self.num_k))

        if solver == 'euler':
            num_steps = int(jnp.ceil((ts[-1] - ts[0]) / dt))
            ts_, xs_, log_path = ma.solve_vector(params, self, omega, x0, y0, ts[0], num_steps, dt, keys[3], args)

            # interpolate for requested timesteps
            xs = jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1)(ts, ts_, xs_)
        else:
            xs, log_path = ma.solve_diffrax(params, self, omega, x0, y0, ts, dt, keys[3], solver, args)
        return xs, log_path


class VideoSDE:
    """
    Latent Video Model.
    Args:
        x0_prior (diffrax.distribution.Distribution): Prior for x0.
        x0_prior_learnable (bool): Whether the prior for x0 is learnable.

    """
    def __init__(
        self,
        image_size,
        num_channels,
        num_features,
        num_latents,
        num_contents,
        x0_prior: distrax.Distribution,
        x0_prior_learnable: bool,
        sde,
    ):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_features = num_features
        self.num_latents = num_latents
        self.num_contents = num_contents
        self._x0_prior = x0_prior
        self.x0_prior_learnable = x0_prior_learnable
        self._encoder = Encoder(image_size, num_channels, num_features, num_features)
        self._decoder = Decoder(image_size, num_channels, num_features, num_latents)
        self._content = Content(num_features, num_contents, num_contents)
        self._infer = Infer(num_features, num_latents)
        self._sde = sde

    def init(self, key):
        keys = jax.random.split(key, 4)
        params = {}

        if self.x0_prior_learnable:
            params['x0_prior'] = self._x0_prior

        dummy_num_timesteps = 5
        params['encoder'] = self._encoder.init(keys[0], jnp.zeros((self.image_size, self.image_size, self.num_channels)))
        params['content'] = self._content.init(keys[1], jnp.zeros((dummy_num_timesteps, self.num_features)))
        params['infer'] = self._infer.init(keys[2], jnp.zeros((dummy_num_timesteps, self.num_features)))
        params['sde'] = self._sde.init(keys[3])
        params['decoder'] = self._decoder.init(keys[1], jnp.zeros((self.num_contents + self.num_latents)))
        return params

    def x0_prior(self, params):
        if self.x0_prior_learnable:
            return params['x0_prior']
        else:
            return self._x0_prior

    def encoder(self, params, *args):
        return self._encoder.apply(params['encoder'], *args)

    def decoder(self, params, *args):
        return self._decoder.apply(params['decoder'], *args)

    def content(self, params ,*args):
        return self._content.apply(params['content'], *args)

    def infer(self, params, *args):
        return self._infer.apply(params['infer'], *args)

    def sde(self, params, *args):
        return self._sde(params['sde'], *args)

    def __call__(self, params, key, ts, frames, dt, solver):
        keys = jax.random.split(key, 2)
        num_frames, height, width, num_channels = frames.shape

        x0_prior = self.x0_prior(params)
        h = self.encoder(params, frames)
        w = self.content(params, h)
        x0_posterior, h = self.infer(params, h)
        context = {'ts': ts, 'hs': h}
        x0 = x0_posterior.sample(seed=keys[0])
        kl_x0 = x0_posterior.kl_divergence(x0_prior)

        xs, logpath = self.sde(params, keys[1], x0, ts, dt, solver, {'context': context})
    
        frames_ = self.decoder(params, jnp.concatenate([w[None, :].repeat(len(xs), axis=0), xs], axis=-1))
        return frames_, (kl_x0, logpath)
