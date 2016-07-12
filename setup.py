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
    name="PyQPECgen",
    version="0.1.0",
    description=qpecgen.__doc__,
    author="Trish Gillett-Kawamoto",
    url="https://github.com/discardthree/PyQPECgen",
    packages=find_packages(exclude=["tests"]),
    install_requires=_load_requires_from_file("requirements.txt"),
    test_suite="tests.suite"
)
