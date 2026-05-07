
# HANNA
[![Pre-Print](https://img.shields.io/badge/Paper-Available-brightgreen)](https://www.nature.com/articles/s41467-026-71430-y)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Website MLPROP](https://img.shields.io/badge/Website-MLPROP-darkred?logo=https://ml-prop.mv.rptu.de/img/Logo.png&labelColor=4f4f4f)](https://ml-prop.mv.rptu.de/)



<p align="center">
  <img src="img/github_HANNA.png" alt="HANNA Overview" width="800"/>
</p>

HANNA is a machine learning model for predicting the excess Gibbs energy of the liquid phase in mixtures with an arbitrary number of components. Through automatic differentiation, HANNA derives thermodynamically consistent activity coefficients from the excess Gibbs energy. HANNA is trained to more than 800,000 experimental data points of vapor-liquid equilibria, liquid-liquid equilibria, activity coefficients at infinite dilution and excess enthalpies in binary mixtures. As input, only the SMILES notation of all molecules and the considered state point (composition and temperature) is required. In comprehensive benchmarks, HANNA was found to yield better results than the state-of-the-art models of the UNIFAC family and other ML models.

**Note**: This repository is based on the HANNA prototype implementation by [tspecht93](https://github.com/tspecht93/HANNA), which was restricted to binary mixtures and not trained on liquid-liquid equilibrium data or excess enthalpies; the corresponding original paper is available [here](https://pubs.rsc.org/en/Content/ArticleLanding/2024/SC/D4SC05115G).


### Easy Use
You can explore HANNA and other models from our working group on our interactive web interface, [MLPROP](https://ml-prop.mv.rptu.de/), without any installation.
<p align="center">
  <a href="https://ml-prop.mv.rptu.de/">
    <img src="img/MLPROP_logo.png" alt="TOC Figure"/>
  </a>
</p>

## Installing and using HANNA

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/marco-hoffmann/HANNA.git
   cd HANNA
   ```

2. **Create the conda environment:**

   Use the provided `.yml` file to create the conda environment.

   ```bash
   conda env create -f environment.yml
   ```

   Note: If you want to run the model on a GPU, make sure to install the PyTorch version with CUDA support.

3. **Test the installation:**

   Open the `demo.ipynb` notebook and run the cells using the HANNA environment to test the installation.

   If you obtain output plots similar to the following, the installation was successful.

   **Binary Prediction:**

   <p align="center">
     <img src="img/binary_output.png" alt="HANNA binary demo output" width="900"/>
   </p>

   **Ternary Prediction:**

   <p align="center">
     <img src="img/ternary_output.png" alt="HANNA ternary demo output" width="900"/>
   </p>


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Tested Versions

The software has been tested with the following package versions:

- python==3.10.19
- pytorch==2.10.0
- numpy==2.2.6
- pandas==2.3.3
- rdkit==2025.9.5
- transformers==5.1.0
- tokenizers==0.22.2
- ipykernel==7.2.0
- ipywidgets==8.1.8
- scikit-learn==1.7.2
- matplotlib==3.10.8
