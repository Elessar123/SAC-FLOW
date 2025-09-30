### 0. About

This document records all the steps needed to install packages related to reproducing the results in `SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling`. We borrowed the installation process of [ReinFlow](https://github.com/ReinFlow/ReinFlow), [cleanrl](https://github.com/vwxyzjn/cleanrl), and [QC-FQL](https://github.com/ColinQiyangLi/qc).


### 1. Configure pip source, conda source channels, and create environment
Create the environment.

```bash
# create environment
conda create -n sacflow python=3.10 -y
conda activate sacflow
# rename your repo folder as ReinFlow
# (omitted)
```


### 2. Setup Environment for OpenAI Gym
Here we list one possible approach to install mujoco_py. 

```bash
# make mujoco directory
mkdir $HOME/.mujoco
cd ~/.mujoco 
# download mujoco210 and mujoco-py from this source or other sources
https://drive.google.com/drive/folders/15fcrWlTCwFxZkxpMPSrcNzCfTxjj8WE4?usp=sharing
# unzip them to your root directory
unzip ./mujoco210.zip
unzip ./mujoco-py.zip
# add link to mujoco
echo -e '# link to mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
export PATH="$LD_LIBRARY_PATH:$PATH" ' >> ~/.bashrc
source ~/.bashrc
conda activate reinflow
# install cython
pip install 'cython<3.0.0' -i https://pypi.tuna.tsinghua.edu.cn/simple

# if you don't have root privilege or cannot update the driver (e.g. in a container)
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
echo -e 'CPATH=$CONDA_PREFIX/include' >> ~/.bashrc
source ~/.bashrc
# else, if you have root privilege: 
sudo apt-get install patchelf
sudo apt-get install python3-dev build-essential libssl-dev libffi-dev libxml2-dev  
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

# install mujoco-py
cd ~/.mujoco/mujoco-py
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -e . --no-cache -i https://pypi.tuna.tsinghua.edu.cn/simple
# test mujoco-py installation 
python3
import mujoco_py # there should be no import errors. 
dir(mujoco_py)   # you should see a lot of methods if you successfully installed mujoco_py. 
```

#### [Debug Helper] If you don't have root privilege, or meet the error '#include <GL/glew.h>'...
```bash
    4 | #include <GL/glew.h>
      |          ^~~~~~~~~~~
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```
You can still install mujoco without using sudo commands by the following bash commands, according to https://github.com/openai/mujoco-py/issues/627
```bash
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
echo -e 'CPATH=$CONDA_PREFIX/include' >> ~/.bashrc
source ~/.bashrc
```

#### [Debug Helper] If you meet this error: version `GLIBCXX_3.4.30‘ not found...
```bash
# link GLIBCXX_3.4.30. reference: https://blog.csdn.net/L0_L0/article/details/129469593
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX # 1. first check if the thing exists: 
# you should be ablt to see GLIBCXX_3.4 ~ GLIBCXX_3.4.30
cd <PATH_TO_YOUR_ANACONDA3>/envs/reinflow/bin/../lib/ # 2. then create soft links
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```

### 3.Install other packages
```bash
# install reinflow package
conda activate sacflow
pip install -r requirements.txt
```

### 4.Install jax
```bash
# install reinflow package
pip install --upgrade "jax[cuda12]"
pip install flax==0.10.7 distrax==0.1.5
```

### 5. Setup environment for Robomimic and OGBench
#### (a) OGBench
```bash
pip install dm_control==1.0.28 ml_collections==1.1.0 ogbench==1.1.2 torchvision==0.19.1
```
#### (b) Robomimic
```bash
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .

```

#### [Debug Helper] if you see `jax.random.key` errrors:
Just degrade the diffusers, use 
```bash
pip install diffusers==0.21.4
```


### 6. Dataset (for offline-to-online training, same as QC-FQL)
For robomimic, we assume the datasets are located at ~/.robomimic/lift/mh/low_dim_v15.hdf5, ~/.robomimic/can/mh/low_dim_v15.hdf5, and ~/.robomimic/square/mh/low_dim_v15.hdf5. The datasets can be downloaded from https://robomimic.github.io/docs/datasets/robomimic_v0.1.html (under Method 2: Using Direct Download Links - Multi-Human (MH)). 

For cube-quadruple, we use the 100M-size offline dataset. It can be downloaded from https://github.com/seohongpark/horizon-reduction via

```bash
wget -r -np -nH --cut-dirs=2 -A "*.npz" https://rail.eecs.berkeley.edu/datasets/ogbench/cube-quadruple-play-100m-v0/
```
and include this flag in the command line --ogbench_dataset_dir=[realpath/to/your/cube-quadruple-play-100m-v0/] to make sure it is using the 100M-size dataset.