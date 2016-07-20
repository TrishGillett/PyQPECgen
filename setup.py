""" Package information.
"""
from setuptools import setup, find_packages
import qpecgen

def _load_requires_from_file(filepath):
    """ Read a package list from a given file path.

    Args:
      filepath: file path of the package list.

    Returns:
      a list of package names.
    """
    with open(filepath) as fp:
        return [pkg_name.strip() for pkg_name in fp.readlines()]


setup(
    name="pyqpecgen",
    version="0.1.0",
    description="A QPEC problem generator for Python",
    long_description=(
        "This package provides a Python translation of the Matlab package "
        "Qpecgen which generates random MPEC test problems with quadratic "
        "objective functions and affine variational inequality constraints. "
        "For more information, see the paper and accompanying code by "
        "Houyuan Jiang, Daniel Ralph, 1997."),
    author="Trish Gillett-Kawamoto",
    author_email="discardthree@gmail.com",
    url="https://github.com/discardthree/PyQPECgen",
    packages=find_packages(exclude=["tests"]),
    install_requires=_load_requires_from_file("requirements.txt"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    test_suite='tests.suite',
    license="MIT"
)
