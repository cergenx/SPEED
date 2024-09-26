# setup.py
from setuptools import setup, find_packages

setup(
    name='speed',
    version='0.1.2',
    packages=find_packages(include=['speed', 'speed.*']),
    install_requires=[
        'numpy>=1.24.3',
        'scipy>=1.10.1',
        'statsmodels>=0.14.1',
        'pandas>=2.0.1',
        'matplotlib>=3.7.1',
        'seaborn>=0.12.2',
        'scikit-learn>=1.2.2',
        'pytest>=7.3.1',
    ],
)
