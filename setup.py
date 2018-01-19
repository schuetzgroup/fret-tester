import os
import re

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# read version from sdt._version
vfile = os.path.join("fret_tester", "_version.py")
vstr = read(vfile)
vre = r"^__version__ = ['\"]([^'\"]*)['\"]"
match = re.search(vre, vstr, re.M)
if match:
    vstr = match.group(1)
else:
    raise RuntimeError("Unable to find version in " + vfile)


setup(
    name="fret-tester",
    version=vstr,
    description="Analyze smFRET kinetics using comprehensive MC simulations "
                "and testing",
    author="Lukas Schrangl",
    author_email="lukas.schrangl@tuwien.ac.at",
    url="https://github.com/schuetzgroup/fret-tester",
    license="GPLv3+",
    install_requires=["numpy",
                      "scipy",
                      "matplotlib"],
    packages=find_packages(include=["fret_tester*"]),
    long_description=read("README.md")
)
