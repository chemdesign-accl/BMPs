# BMPs

**BMPs** is a modular Graph Neural Network (GNN) framework designed for molecular property prediction built on the Metalayer[1] framework1. It supports multiple node-block models having in common a bidirectional message-passing. Options include attention mechanism applied to the message (similar to GAT), convolutional normalization, skip or include raw nodes connections.

MoleculeNet[2] (BACE and BBBP) and BindingDB[3] (TRPA1) datasets are included in this repository

Node-Models	Description
BMP	        Base message passing without self-nodes
ABMP	        Attention-enhanced message passing
CBMP          	Convolutional-normalization message passing
BMP+SN		BMP with self-nodes
ABMP+SN		ABMP with self-nodes

**INSTALLATION**

PyTorch Geometric (PyG) and its extensions require specific installation steps due to version-specific CUDA bindings. Follow the tested working environment setup below for best results using python 3.11 with GPU implementation using CUDA 12.6 and currently supported torch version (2.6). *These versions reflect the latest tested configuration. In future releases, PyG may support newer versions of PyTorch. Please refer to the official PyG installation guide to adapt accordingly.

1. Create and activate a clean conda environment:

    bash conda remove --name torch_gpu --all -y conda create -n torch_gpu python=3.11 -y conda activate torch_gpu 
    
2. Install PyTorch with CUDA 12.6:

    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
    
or with CPU-only pytorch:

    pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
    
3. Install specific CUDA libraries manually (if required):

    pip install --force-reinstall nvidia-cusparse-cu12==12.5.4.2 nvidia-nvjitlink-cu12==12.6.85 \
    nvidia-cublas-cu12==12.6.4.1 nvidia-cuda-runtime-cu12==12.6.77
    
4. Install PyG extensions (must match your CUDA + PyTorch version):

    pip install pyg-lib torch_scatter torch_sparse torch_cluster torch_spline_conv\
    -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
    
    pip install torch_geometric
    
Finally, install the package:

    pip install git+https://github.com/chemdesign-accl/BMPs.git

or for development install via SSH:

    git clone git@github.com:chemdesign-accl//BMPs.git
    cd BMPs/
    pip install -e .

**External Citations:**

[1] Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., Tacchetti, A., Raposo, D., Santoro, A., Faulkner, R., Gulcehre, C., Song, F., Ballard, A., Gilmer, J., Dahl, G., Vaswani, A., Allen, K., Nash, C., Langston, V., Dyer, C., Heess, N., Wierstra, D., Kohli, P., Botvinick, M., Vinyals, O., Li, Y., & Pascanu, R. (2018). Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261.

[2] Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. Chemical Science, 9(2), 513–530.

[3] Gilson, M. K., Liu, T., Baitaluk, M., Nicola, G., Hwang, L., & Chong, J. (2016). BindingDB in 2015: a public database for medicinal chemistry, computational chemistry and systems pharmacology. Nucleic Acids Research, 44(D1), D1045–D1053.
