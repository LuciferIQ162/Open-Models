{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f42b8d7b-bb1b-41c2-be2c-2f5a57e992d0",
   "metadata": {},
   "source": [
    "# Robot Skill compiler"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1b9b22d2-e6c5-447b-9f9b-34285d68facd",
   "metadata": {},
   "source": [
    "## Project thesis: Build a compiler-like pipeline that converts natural-language goals into a verified intermediate robot program,\n",
    "## then into executable task and motion plans."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba976029-427d-4302-9995-7837036da727",
   "metadata": {},
   "source": [
    "### Phase -1 define any tiny robot domain: navigation, pickup, place, inspect, wait, recover."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7500f964-5849-4a08-8d38-eb8bdbe8e729",
   "metadata": {},
   "source": [
    "# In this we will demonstrate monte carlo Quatization in here"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "255b8b92-43e0-4116-90b9-ce6696e2577e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/lucy/.venv/bin/python3.10\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "print(sys.executable)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f29a3bca-be42-4485-97bd-1ff6cc13da14",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: torchvision in /home/lucy/.venv/lib/python3.10/site-packages (0.28.0)\n",
      "Requirement already satisfied: numpy in /home/lucy/.venv/lib/python3.10/site-packages (from torchvision) (2.2.6)\n",
      "Requirement already satisfied: torch==2.13.0 in /home/lucy/.venv/lib/python3.10/site-packages (from torchvision) (2.13.0)\n",
      "Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /home/lucy/.venv/lib/python3.10/site-packages (from torchvision) (12.3.0)\n",
      "Requirement already satisfied: filelock in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (3.32.2)\n",
      "Requirement already satisfied: typing-extensions>=4.10.0 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (4.16.0)\n",
      "Requirement already satisfied: setuptools>=77.0.3 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (79.0.1)\n",
      "Requirement already satisfied: sympy>=1.13.3 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (1.14.0)\n",
      "Requirement already satisfied: networkx>=2.5.1 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (3.4.2)\n",
      "Requirement already satisfied: jinja2 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (3.1.6)\n",
      "Requirement already satisfied: fsspec>=0.8.5 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (2026.7.0)\n",
      "Requirement already satisfied: cuda-toolkit==13.0.3 in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (13.0.3.0)\n",
      "Requirement already satisfied: cuda-bindings<14,>=13.0.3 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (13.3.1)\n",
      "Requirement already satisfied: nvidia-cudnn-cu13==9.20.0.48 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (9.20.0.48)\n",
      "Requirement already satisfied: nvidia-cusparselt-cu13==0.8.1 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (0.8.1)\n",
      "Requirement already satisfied: nvidia-nccl-cu13==2.29.7 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (2.29.7)\n",
      "Requirement already satisfied: nvidia-nvshmem-cu13==3.4.5 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (3.4.5)\n",
      "Requirement already satisfied: triton==3.7.1 in /home/lucy/.venv/lib/python3.10/site-packages (from torch==2.13.0->torchvision) (3.7.1)\n",
      "Requirement already satisfied: nvidia-curand==10.4.0.35.* in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (10.4.0.35)\n",
      "Requirement already satisfied: nvidia-nvjitlink<14,>=13.0.88 in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (13.3.33)\n",
      "Requirement already satisfied: nvidia-cufft==12.0.0.61.* in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (12.0.0.61)\n",
      "Requirement already satisfied: nvidia-cufile==1.15.1.6.* in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (1.15.1.6)\n",
      "Requirement already satisfied: nvidia-cuda-cupti==13.0.85.* in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (13.0.85)\n",
      "Requirement already satisfied: nvidia-cusparse==12.6.3.3.* in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (12.6.3.3)\n",
      "Requirement already satisfied: nvidia-nvtx==13.0.85.* in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (13.0.85)\n",
      "Requirement already satisfied: nvidia-cuda-runtime==13.0.96.* in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (13.0.96)\n",
      "Requirement already satisfied: nvidia-cublas==13.1.1.3.* in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (13.1.1.3)\n",
      "Requirement already satisfied: nvidia-cusolver==12.0.4.66.* in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (12.0.4.66)\n",
      "Requirement already satisfied: nvidia-cuda-nvrtc==13.0.88.* in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3; platform_system == \"Linux\"->torch==2.13.0->torchvision) (13.0.88)\n",
      "Requirement already satisfied: cuda-pathfinder>=1.4.2 in /home/lucy/.venv/lib/python3.10/site-packages (from cuda-bindings<14,>=13.0.3->torch==2.13.0->torchvision) (1.6.0)\n",
      "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/lucy/.venv/lib/python3.10/site-packages (from sympy>=1.13.3->torch==2.13.0->torchvision) (1.3.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /home/lucy/.venv/lib/python3.10/site-packages (from jinja2->torch==2.13.0->torchvision) (3.0.3)\n"
     ]
    }
   ],
   "source": [
    "!{sys.executable} -m pip install torchvision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2f245f32-6a98-42a1-9456-f9ca0597140d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
      "I0000 00:00:1786293412.960087    5061 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
      "To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
      "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
      "I0000 00:00:1786293415.356141    5061 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torchvision\n",
    "import numpy\n",
    "import pandas\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "2ac12117-9638-4bf0-9f8b-f1341765db48",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Robot:\n",
    "    def __init__(self, action, location):\n",
    "        self.action = action\n",
    "        self.location = location"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb1490d1-e294-472c-b43e-c7f6b24c2eec",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (.venv)",
   "language": "python",
   "name": "myvenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
