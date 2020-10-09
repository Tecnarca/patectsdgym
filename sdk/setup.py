"""Setup file for differential privacy package."""
from setuptools import setup, find_packages
import os
import shutil

_major = "0.1"
_minor = "0.1"

VERSION = "{}.{}".format(_major, _minor)
SELFVERSION = VERSION
if os.path.exists("patch.version"):
    with open("patch.version", "rt") as bf:
        _patch = str(bf.read()).strip()
        SELFVERSION = "{}.{}".format(VERSION, _patch)

DEPENDENCIES = []
EXTRAS = { }

with open("../README.md", "r", encoding="utf-8") as f:
    README = f.read()
inline_license = "MIT"

setup(
    name="patectgan",

    version=SELFVERSION,

    description="",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Tecnarca",
    license=inline_license,
    packages=find_packages(exclude=["*.tests"]),

    install_requires=DEPENDENCIES,

    include_package_data=True,

    extras_require=EXTRAS,

    data_files=[],
    zip_safe=False
)