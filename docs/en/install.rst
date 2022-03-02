Installation
############

MMEditing is a pure Python toolbox and is ready for use from source code.
However, its dependencies, PyTorch and MMCV, take some effort to install.
This page describes some best practices for installing PyTorch, MMCV, and MMEditing.

Prerequisites
*************

* `Python`_ >= 3.7
* `pip`_ and/or `conda`_
* `Git`_
* (Recommended) NVIDIA GPU with driver version >= 440.33 (Linux) or >= 441.22 (Windows)
* (Optional) CUDA and C++ (GCC / Clang / MSVC) compilers if you need to compile `MMCV`_ from source codes


Install CPU Version
===================

MMEditing is fully supported on CPUs, despite very slow running speed.
However, the CPU version is much more lightweight.
So if you just want to perform a quick run, the CPU version is good enough.

Step 1.
Create and activate a conda virtual environment

.. code-block::

  conda create -n mmedit python=3.8 -y
  conda activate mmedit

Step 2.
Install the CPU version PyTorch and torchvision

.. code-block::

  conda install pytorch torchvision cpuonly -c pytorch

Step 3.
Install the CPU version of MMCV

.. code-block::

  pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7/index.html "opencv-python<=4.5.4.60"


```shell
conda install pytorch==1.7.1 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7/index.html "opencv-python<=4.5.4.60"
```

Note 1: Higher version `opencv-python` is not supported.

Note 2: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g. 1` If you have CUDA 10.1 installed under `/usr/local/cuda`, you need to install the prebuilt PyTorch with CUDA 10.1.

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

c. Clone the mmediting repository.

```shell
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
```

d. Install build requirements and then install mmediting.

```shell
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
```

Install CUDA Version
====================

To enable full



.. _Git: https://git-scm.com/
.. _Python: https://www.python.org/
.. _conda: https://docs.conda.io/en/latest/
.. _pip: https://pip.pypa.io/en/stable/
.. _pip: https://pip.pypa.io/en/stable/
.. _MMCV: https://github.com/open-mmlab/mmcv
.. _PyTorch: https://pytorch.org/
.. _CUDA version table: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions
