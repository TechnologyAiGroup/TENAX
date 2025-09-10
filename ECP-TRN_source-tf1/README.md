# Reproducing ECP-TRN


## Folder Structure
The folder structures should be the same as following
```
ECP-TRN_source-tf1
├── ECP_TRN/
├── TRN_Ex/
├── TRN_Ex_LSPUF/
```

### ECP_TRN

ECP-TRN source code, including XOR APUF and LSPUF attack source code, and some data sets

## TRN_Ex

Source code and experimental logs for various XOR APUF attack experiments using ECP-TRN

The 128-7 XOR APUF experiment results in memory overflow. This was performed using Python 3.6 and TensorFlow 1.12.0 for the CPU.

Other reproducible experiments were performed using Python 3.6 and TensorFlow 1.12.0 for the GPU.

## TRN_Ex_LSPUF

Source code and experimental logs for various LSPUF attack experiments using ECP-TRN



## Citation
 ```text
@article{santikellurtensorattack,
   author={P. {Santikellur} and R. S. {Chakraborty}},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={A Computationally Efficient Tensor Regression Network based Modeling Attack on XOR Arbiter PUF and its Variants}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCAD.2020.3032624}
}

@inproceedings{santikellur2019computationally,
  title={A Computationally Efficient Tensor Regression Network based Modeling Attack on XOR APUF},
  author={Santikellur, Pranesh and Prakash, Shashi Ranjan and Chakraborty, Rajat Subhra and others},
  booktitle={2019 Asian Hardware Oriented Security and Trust Symposium (AsianHOST)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
 ```



