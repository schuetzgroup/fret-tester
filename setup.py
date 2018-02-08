# Copyright 2017-2018 Lukas Schrangl
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
