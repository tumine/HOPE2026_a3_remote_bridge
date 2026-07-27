from setuptools import find_packages
from setuptools import setup

setup(
    name='hope_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('hope_msgs', 'hope_msgs.*')),
)
