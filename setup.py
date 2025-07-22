from setuptools import setup, find_packages

setup(
    name='bmpnns',
    version='0.1',
    description='MPNNs for molecular prediction',
    author='Alma C. Castaneda-Leautaud',
    packages=find_packages(),
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
        "torch==2.7.1+cu128",
        "torchvision==0.21.0+cu128",
        "torchaudio==2.7.0+cu128",
        "pyg-lib==0.4.0+pt27cu128",
        "torch-scatter==2.1.2+pt27cu128",
        "torch-sparse==0.6.18+pt27cu128",
        "torch-cluster==1.6.3+pt27cu128",
        "torch-spline-conv==1.2.2+pt27cu128"
    ],
    python_requires='>=3.11'
)

