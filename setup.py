#!/usr/bin/env python3

from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="blockcrafter",
    version="1.0",
    description="Description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mapcrafter/blockcrafter",
    author="m0r13",
    author_email="m0r13@mapcrafter.org",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",

        "Topic :: Multimedia :: Graphics :: 3D Rendering",
    ],
    keywords="",
    packages=find_packages(),
    install_requires=["numpy", "vispy", "Pillow"],
    extras_require={},
    package_data={
        "blockcrafter" : ["custom_assets", "blockstates.properties"]
    },
    include_package_data=True,
    entry_points={
        "console_scripts" : [
            "blockcrafter-export=blockcrafter.export:main",
            "blockcrafter-visualize=blockcrafter.visualize:main"
        ]
    },
    project_urls={
        "Mapcrafter" : "https://mapcrafter.org"
    },
)
