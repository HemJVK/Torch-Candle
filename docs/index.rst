🕯️ Torch-Candle Documentation
============================

Welcome to the architectural documentation for **Torch-Candle**, the high-performance PyTorch-compatible framework backed by the Rust Candle core engine.

.. toctree::
   :maxdepth: 2
   :caption: Core Architecture:

   c10
   aten
   backends

.. toctree::
   :maxdepth: 2
   :caption: Features:

   multiprocessing
   func
   jit
   distributed

Installation & Building
-----------------------

Torch-Candle features a dynamic hardware dispatch compiler backend. Install via setuptools:

.. code-block:: bash

   # CPU Compilation with custom OpenMP fallbacks
   pip install -e .

   # CUDA-enabled hardware acceleration compilation
   USE_CUDA=1 pip install -e .

   # AMD ROCm backend compilation
   USE_ROCM=1 pip install -e .

Zero-Copy Shared Memory Multiprocessing
---------------------------------------

Torch-Candle overrides the default reduction serialization mechanics for deep process integrations:

.. code-block:: python

   import torch_candle.multiprocessing as mp

   # Share memory zero-copy reduction automatically registers
   x = torch.randn(100, 100)
   x.share_memory_()
   assert x.is_shared()

Distributed Collectives
-----------------------

Torch-Candle supports multi-GPU process scaled synchronization:

.. code-block:: python

   import torch_candle.distributed as dist

   dist.init_process_group("nccl", rank=0, world_size=2)
   x = torch_candle.ones(5)
   dist.all_reduce(x, op="sum")
