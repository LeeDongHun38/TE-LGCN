"""Setup script for TE-LGCN package."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="te-lgcn",
    version="0.1.0",
    description="Topic-Enhanced LightGCN for Recommendation",
    author="TE-LGCN Research Team",
    author_email="",
    url="https://github.com/yourusername/topic-enhanced-lightgcn",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="recommendation-systems graph-neural-networks collaborative-filtering",
)
