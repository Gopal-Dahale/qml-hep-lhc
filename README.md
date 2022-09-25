<div align="center">
  
  # Quantum Convolutional Neural Networks<br>for High-Energy Physics Analysis at the LHC

A Google Summer of Code 2022 Project Repository.<br>
The goal of this study is to show the capabilities of QML especially QCNN for classifying the HEP image datasets.
</div>

## Table of (Main) Contents
- [Introduction](#introduction)
  - [Synopsis](#synopsis)
- [Usage](#usage)
  - [Code Description](#code-description)
  - [Installation](#installation)
  - [Documentation Tutorials and Development](#documentation-tutorials-and-development)
- [Project's Datasets](#projects-datasets)
- [Research](#research)
  - [Results](#results)
- [References](#references)

## Introduction

- **Organization**
  - [Machine Learning for Science (ML4Sci)](https://ml4sci.org/)
- **Contributor**
  - [Gopal Ramesh Dahale](https://www.linkedin.com/in/gopal-ramesh-dahale-7a3087198/)
- **Mentors**
  - [Prof. Sergei V. Gleyzer](http://sergeigleyzer.com/), [Dr. Emanuele Usai](https://orcid.org/0000-0001-9323-2107), and [Raphael Koh](https://www.raphaelkoh.me/)
- **Project Details**
  - [Project](https://ml4sci.org/gsoc/2022/proposal_QMLHEP2.html)
  - [Proposal](https://ml4sci.org/gsoc/2022/proposal_QMLHEP2.html)
  - [GSoC Project Page](https://summerofcode.withgoogle.com/programs/2022/projects/0gbpQgKv)


### Synopsis
Determining whether an image of a jet particle corresponds to signals or background signals is one of the many challenges faced in High Energy Physics. CNNs have been effective against jet particle images as well for classification purposes. Quantum computing is promising in this regard and as the QML field is evolving, this project aims to understand and implement QCNN and gain some enhancement.

The goal of this study is to show the capabilities of QML especially QCNN for classifying the HEP image datasets. QCNN can be completely quantum or can be a hybrid with classical. The aim is to implement both. We will use quantum variational classification instead of the final FC classical layers in the quantum setting. This will give more depth about the quantum power that can be used in the near term future.


## How to Use

### Code Description
The repository contains [TensorFlow Quantum](https://www.tensorflow.org/quantum) implementation of quantum convolution and classifier with various data encoding schemes and ansatzes including data reuploading. Models in JAX and Pennylane are also added as they have significant speed up during the training. Hybrid as well as fully quantum models can be created using the layers implemented. JAX models can be trained on TPUs as well.

### Installation

Tested on Ubuntu 22.04.1 LTS
```
git clone https://github.com/Gopal-Dahale/qml-hep-lhc.git
cd qml-hep-lhc
python -m venv qenv
source qenv/bin/activate
export PYTHONPATH=.
pip install -r requirements.txt
```

### Documentation Tutorials and Developement
- Documentation: In progress
- Tutorials: Update
- Development notebooks: Update

