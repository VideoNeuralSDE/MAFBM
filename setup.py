from setuptools import setup, find_packages

setup(
    name="video-sde",
    version="0.0.1",
    install_requires=[
        # 'torch',
        # 'torchvision',
        'scipy',
        'numpy',
        'jupyter',
        # 'pytorch-lightning',
        'matplotlib',
        'wandb',
        'stochastic',
        'moviepy',
        'imageio',
        'jsonargparse',
        'flm @ git+ssh://git@github.com/bencoscia/flm.git',
        'flax',
        'optax',
        'diffrax',
        'distrax @ git+https://github.com/google-deepmind/distrax@master',
        'pandas',
        'seaborn'
    ],
    packages=find_packages(include=["sde"]),
)