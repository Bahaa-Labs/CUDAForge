import os
import shutil
from setuptools import setup, find_packages

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name="cudaforge",
    version="0.1.0",
    packages=find_packages(include=["cudaforge*"]),
    include_package_data=True,
    zip_safe=False,
)