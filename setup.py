#!/usr/bin/env python3
from os.path import dirname, realpath
from setuptools import find_packages, setup, Distribution
import setuptools.command.build_ext as _build_ext


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


class build_ext(_build_ext.build_ext):
    def run(self):
        pass


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name='flow',
    version='0.1.0',
    distclass=BinaryDistribution,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    install_requires=_read_requirements_file(),
    # install_requires=['tensorflow'],
    # dependency_links = ['git+ssh://github.com/openai/gym.git'],
    zip_safe=False,
)
