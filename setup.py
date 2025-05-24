from setuptools import find_packages, setup

setup(
  name='options',
  version='1.0.0',
  packages=find_packages(),
  entrypoints={
    'console_scripts': [
      'foo=src.main:main',
    ],
  },
)