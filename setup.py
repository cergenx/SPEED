# setup.py
from setuptools import setup, find_packages

setup(
    name='speed',
    version='0.1',
    packages=find_packages(include=['speed', 'speed.*']),
)
