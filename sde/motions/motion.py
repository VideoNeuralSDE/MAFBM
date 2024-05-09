import math
import numpy as np
import torch
from stochastic.processes.continuous import BrownianMotion, FractionalBrownianMotion
from scipy.stats import levy_stable
import flm  # https://flm.readthedocs.io/en/latest/autosummary/flm.flm.html


class Motion:
    def __init__(self, motion_type, shape, t0, t1, num_steps, scale=1., **kwargs):
        self.shape = shape
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.num_steps = num_steps
        if motion_type == 'zero':
            self.motion = torch.zeros((num_steps + 1, *shape))
        elif motion_type == 'brownian':
            bm = BrownianMotion(t=float(t1 - t0), scale=scale)
            num_paths = math.prod(self.shape)
            motion = np.stack([bm.sample(num_steps) for _ in range(num_paths)], axis=-1)
            motion = motion.astype(np.float32)
            motion = motion.reshape((num_steps + 1, *shape))
            self.motion = torch.from_numpy(motion)
        elif motion_type == 'fractional_brownian':
            bm = FractionalBrownianMotion(hurst=kwargs['hurst'], t=float(t1 - t0))
            num_paths = math.prod(self.shape)
            motion = np.stack([bm.sample(num_steps) * scale for _ in range(num_paths)], axis=-1)
            motion = motion.astype(np.float32)
            motion = motion.reshape((num_steps + 1, *shape))
            self.motion = torch.from_numpy(motion)
        elif motion_type == 'levy':
            num_paths = math.prod(self.shape)
            motion = [np.zeros(num_paths, dtype=np.float32)]
            for _ in range(num_steps):
                motion.append(motion[-1] + self.dt ** (1. / kwargs['alpha']) * levy_stable.rvs(kwargs['alpha'], kwargs['beta'], size=num_paths, scale=scale).astype(np.float32))
            motion = np.stack(motion)
            motion = motion.reshape((num_steps + 1, *shape))
            self.motion = torch.from_numpy(motion)
        elif motion_type == 'fractional_levy':
            fractional_levy_motion = flm.FLM(num_steps + 1, kwargs['hurst'], kwargs['alpha'], scale=scale)
            num_paths = math.prod(self.shape)
            fractional_levy_motion.generate_realizations(num_paths, progress=False)
            motion = fractional_levy_motion.realizations.T.astype(np.float32)
            motion = motion.reshape((num_steps + 1, *shape))
            self.motion = torch.from_numpy(motion)
        else:
            raise NotImplementedError(f'Motion type {motion_type} is not implemented.')

    @property
    def dt(self):
        return (self.t1 - self.t0) / (self.num_steps + 1)

    def sample(self, t):
        assert self.t0 <= t <= self.t1
        i = (t - self.t0) / (self.t1 - self.t0) * (self.num_steps + 1)
        if torch.is_tensor(i):
            i = i.to(torch.int64)
            return self.motion[i].to(i.device)
        else:
            i = int(i)
            return self.motion[i]