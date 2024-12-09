# setup.py
from setuptools import setup, find_packages

# Read requirements.txt, ignoring comments
def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="dental_segmentation",
    version="0.1",
    install_requires=read_requirements(),
    package_dir={"": "src"},  
    packages=find_packages(where="src"),  
)