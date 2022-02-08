from setuptools import setup

from src.mirle_vision import __version__

setup(
    name='mirle_vision',
    version=__version__[1:],
    packages=['mirle_vision'],
    package_dir={'': 'src'}
)
