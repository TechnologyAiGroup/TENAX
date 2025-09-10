# TENAX Project

This repository provides the experimental source code of the paper "TENAX", partial FPGA data, experimental settings of TENAX-related methods, and experiment logs.

## File Description

The folder **Datasets** stores automatically generated datasets, FPGA datasets, and code files related to data processing.

The folder **ECP-TRN_source-tf1** contains the source code of ECP-TRN, along with experiment logs obtained by running this code on our generated datasets.

The folder **PUF** stores the code of different PUF classes.

The folder **Save** stores experiment logs generated during the runs.

**TRL.py** defines the TRL layer structure used in TENAX.

**AttackWithPca_main.py** is the source code that uses TENAX or C2D2 to attack XOR APUFs. The dataset will be automatically generated according to the random seed.  
**AttackWithPca_main_128_7.py** and **AttackWithPca_main_256_5.py** are the codes for attacking 128-bit 7-XOR APUF and 256-bit 5-XOR APUF with TENAX, respectively. Due to the larger datasets, some implementation details differ from *AttackWithPca_main.py*, but the core logic remains unchanged.  
Similarly, **AttackWithPca_main_128_7_C2D2.py** and **AttackWithPca_main_256_5_C2D2.py** use C2D2 to attack 128-bit 7-XOR APUF and 256-bit 5-XOR APUF, respectively.

**AttackWithPca_III_main.py** is the source code for attacking XOR APUFs using TENAX(RS).

**main.py** is the source code for attacking XOR APUFs using TENAX(W/O) or LR.  
**main_128_7_LR.py** and **main_256_5_LR.py** are the codes for attacking 128-bit 7-XOR APUF and 256-bit 5-XOR APUF with LR, respectively. Due to larger datasets, some implementation details differ from *main.py*, but the core logic remains unchanged.

**AttackLSPUF_main.py** and **BreakingLSPUF.py** are the source codes for attacking LSPUFs with TENAX(W/O) and BreakingLSPUF, respectively. The dataset will be automatically generated according to the random seed.

**FPGA_AttackWithPca.py** is the source code for attacking XOR APUFs with TENAX or C2D2 using FPGA datasets. The dataset is retrieved from *Datasets* based on the filename. If you want to use TENAX or C2D2 on your own dataset, this code can be adapted.  
**FPGA_main.py**, **FPGA_AttackLSPUF.py**, and **FPGA_BreakingLSPUF.py** are similar to their counterparts without the *FPGA_* prefix, except for the dataset source.

## Environment Requirements

- Python 3.6  
- TensorFlow 2.10 GPU version  
- scikit-learn  
- pandas  

## Quick Start

### 1. Attack XOR APUF

- Run a simple attack using TENAX:

```bash
python AttackWithPca_main.py
```

- To use C2D2 instead, modify

```python
method = "Ett"
```

to

```python
method = "LR"
```

and run:

```bash
python AttackWithPca_main.py
```

- Run a simple attack using TENAX(W/O):

```bash
python main.py
```

- To use LR instead, modify

```python
method = "Ett"
```

to

```python
method = "LR"
```

and run:

```bash
python main.py
```

- Run a simple attack using TENAX(RS):

```bash
python AttackWithPca_III_main.py
```

All of the above attacks will automatically generate datasets. If you want to test on your own dataset, you can use the corresponding *FPGA_* prefixed version of the code. For example, to use TENAX on your dataset, simply replace

```python
filename = f"./Datasets/XOR_PUF_64_5_CRPs_transformed.csv"
```

with

```python
filename = f"./Datasets/YOURS_XOR_PUF_CRPs_transformed.csv"
```

and run:

```bash
python FPGA_AttackWithPca.py
```

### 2. Attack iPUF

Open the **Save/MXPUF/** folder, where you will find attack source codes for different iPUF configurations, including both TENAX and Split iPUF. Copy the desired code file into the **TENAX/** folder and run:

```bash
python AttackIPUFWithPca_main_1_5.py
```

to run a simple attack on a 64-bit (1,5)-iPUF using TENAX.

Alternatively, run:

```bash
python SplitIPUF_1_5.py
```

to run a simple attack on a 64-bit (1,5)-iPUF using Split iPUF.

Other iPUF configurations follow the same procedure.

### 3. Attack LSPUF

- Run a simple attack using TENAX(W/O):

```bash
python AttackLSPUF_main.py
```

- Run a simple attack using BreakingLSPUF:

```bash
python BreakingLSPUF.py
```





