# EdgeDRNN
This repo mainly contains:
- Training code of DeltaGRU using the dataset from [EdgeDRNN-AMPRO](https://arxiv.org/abs/2002.03197)
- SystemVerilog HDL code of [EdgeDRNN](https://arxiv.org/abs/2012.13600)
- Xilinx SDK Bare-Metal C code for controlling EdgeDRNN on AVNET [MiniZed](https://www.avnet.com/wps/portal/us/products/avnet-boards/avnet-board-families/minized/) 

# Supported Dataset
```
.
└──  hdl                   # SystemVerilog HDL code of EdgeDRNN
    ├── tb                 # Testbench and stimuli
└── python                 # Pytorch Training Code
    ├── data               # AMPRO Walking Dataset
    ├── modules            # Pytorch Modules
    ├── nnlayers           # Pytorch NN Layers
    └── steps              # Pytorch training steps (Pretrain, Retrain, Export)
└── vivado                 # Xilinx Vivado Projects
    └── boardfile          # Boardfile for MiniZed (or add your own board here)
```

# Prerequisite
This project replies on [PyTorch Lightning](https://www.pytorchlightning.ai/) and was tested in Ubuntu 20.04 LTS.

Install Miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Create an environment using the following command:
```
conda create -n pt python=3.8 numpy matplotlib pandas scipy \
    pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Activate the environment.
```
conda activate pt
```

Install the nightly-built PyTorch Lightning.
```
pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip
```

#  Training a DeltaGRU
DeltaGRU can be trained from scratch or by following a pretrain-retrain scheme. The code under `./python` how to train a DeltaGRU on AMPRO dataset using PyTorch Lightning and finally export the parameters of the network and SystemVerilog testbench stimuli. To run the code, navigate to `./python` in your terminal and run the following command:
```
conda activivate pt
python main.py --step pretrain --run_throughput 1
```
To make it faster for you to run the code and functional simulation, the default python code trains a tiny 2L-16H-DeltaGRU network. If you want to change the network size or other hyperparameters, modify `./python/project.py`. After the training and export is done, there will be three extra folders created: `./python/logs/`, `./python/save/` and `./python/sdk/`.
- `./python/logs/` contains the logged metrics during training.
- `./python/save/` contains the saved models during training.
- `./python/sdk/` contains the exported model parameters in Xilinx SDK Bare-Metal C libraries.

Moreover, the python code also exports SystemVerilog testbench stimuli to `./hdl/tb/`.

#  Functional Simulation & MiniZed Testing
- Please download our [Example Vivado Project](hehe) and extracted under the `./vivado/` folder.
- Use Xilinx Vivado 2018.2
- All the source files in the Vivado project are embedded inside the project folder to make sure it runs seemlessly on all machines. 
If you update the source code, please make sure to update the Vivado project accordingly (overwrite source files inside the project folder. You can do this simply in Vivado GUI).
- Before running the functional simulation, make sure to define `SIM_DEBUG` in `hdr_macros.v`.
- Before synthesizing the code, make sure to undefine `SIM_DEBUG` in `hdr_macros.v`.
- To test EdgeDRNN on MiniZed Board, open Xilinx SDK in Vivado from `File->Launch SDK`. Before you connect the MiniZed board to your PC, make sure the [Xilinx Cable Drive](https://digilent.com/reference/programmable-logic/guides/install-cable-drivers) is correctly installed. To launch the test programme on MiniZed, in Xilinx SDK, right click the project `edgedrnn_test` and click `Run As->Launch on Hardware (GDB)`.