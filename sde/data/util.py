from .moving_mnist import MovingMNIST
from .kth.kth import KTH, KTHTest
from .clouds import Clouds
from .ball import Ball
from .bar import Bar
from .bounce import Bounce


def get(dataset):
    if dataset == 'mmnist':
        data_train = MovingMNIST(train=True, data_root='/data', seq_len=25, num_digits=2, deterministic=False)
        data_val = MovingMNIST(train=False, data_root='/data', seq_len=25, num_digits=2, deterministic=False)
        image_size = 64
        num_channels = 1
        dt = data_train.step_length
    elif dataset == 'ball':
        image_size = 64
        num_channels = 1
        dt = 1 / 30
        data_train = Ball(train=True, sequence_length=50, hurst=0.5, size=image_size, dt=dt)
        data_val = Ball(train=False, sequence_length=50, hurst=0.5, size=image_size, dt=dt)
    elif dataset == 'bar':
        image_size = 64
        num_channels = 1
        dt = 1 / 30
        data_train = Bar(train=True, sequence_length=50, hurst=0.5, size=image_size, dt=dt)
        data_val = Bar(train=False, sequence_length=50, hurst=0.5, size=image_size, dt=dt)
    elif dataset == 'bounce':
        image_size = 64
        num_channels = 1
        dt = 1 / 30
        data_train = Bounce(train=True, sequence_length=50, size=image_size, dt=dt)
        data_val = Bounce(train=False, sequence_length=50, size=image_size, dt=dt)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    dataset_kwargs = {
        'image_size': image_size,
        'num_channels': num_channels,
        'dt': dt,
    }
    return data_train, data_val, dataset_kwargs
