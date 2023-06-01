from setuptools import setup

from src.aibox_vision import __version__

setup(
    name='aibox_vision',
    version=__version__[1:],
    packages=['aibox_vision'],
    package_dir={'': 'src'}
)
