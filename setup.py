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
        "torch==2.6.0+cu126",
        "torchvision==0.21.0+cu126",
        "torchaudio==2.6.0+cu126",
        "pyg-lib==0.4.0+pt26cu126",
        "torch-geometric==2.6.1",
        "torch_scatter==2.1.2",
        "torch_sparse==0.6.18",
        "torch_cluster==1.6.3",
        "torch_spline_conv==1.2.2"
    ],
    python_requires='>=3.11'
)

