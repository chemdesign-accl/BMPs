import os

from setuptools import setup


source_package_dir = "BMPs"

setup(
    name='BMPs',
    version='2.0.0',
    description='Bidirectional message passing neural networks for molecular prediction',
    author='Alma C. Castaneda-Leautaud',
    packages=[
        "BMPs",
        "BMPs.data",
        "BMPs.model",
    ],
    package_dir={
        "BMPs": source_package_dir,
        "BMPs.data": f"{source_package_dir}/data",
        "BMPs.model": f"{source_package_dir}/model",
    },
    install_requires=[
        "python-dateutil",
        "packaging",
        "rdkit",
        "pandas",
        "matplotlib",
        "numpy",
        "tqdm",
        "Pillow",
        "molvs",
        "mendeleev",
        "scikit-learn",
        "torch>=2.7.0",
        "torchvision>=0.21.0",
        "torchaudio>=2.7.0",
        "pyg-lib",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
    ],
    python_requires='>=3.11'
)
