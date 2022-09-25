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
- [Datasets](#datasets)
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
  - [Proposal](https://github.com/Gopal-Dahale/qml-hep-lhc/blob/main/QCNN%20Proposal.pdf)
  - [GSoC Project Page](https://summerofcode.withgoogle.com/programs/2022/projects/0gbpQgKv)


### Synopsis
Determining whether an image of a jet particle corresponds to signals or background signals is one of the many challenges faced in High Energy Physics. CNNs have been effective against jet particle images as well for classification purposes. Quantum computing is promising in this regard and as the QML field is evolving, this project aims to understand and implement QCNN and gain some enhancement.

The goal of this study is to show the capabilities of QML especially QCNN for classifying the HEP image datasets. QCNN can be completely quantum or can be a hybrid with classical. The aim is to implement both. We will use quantum variational classification instead of the final FC classical layers in the quantum setting. This will give more depth about the quantum power that can be used in the near term future.


## How to Use

### Code Description
The repository contains [TensorFlow Quantum](https://www.tensorflow.org/quantum) implementation of quantum convolution and classifier with various data encoding schemes and ansatzes including data reuploading. Models in [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) and [Pennylane](https://pennylane.ai/) are also added as they have significant speed up during the training. Hybrid as well as fully quantum models can be created using the layers implemented. JAX models can be trained on TPUs as well.

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
- Documentation: Work in progress.
- Tutorials: [Link](https://github.com/Gopal-Dahale/qml-hep-lhc/tree/main/notebooks/Tutorials). Work in progress.
- Development notebooks: [Link](https://github.com/Gopal-Dahale/qml-hep-lhc/tree/main/notebooks/Dev). These notebooks were used during the period of GSoC.

## Datasets

### MNIST

<p align="middle">
  <img height="300 px" src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" title="MNIST Dataset" /> <br>
  <a>MNIST Dataset</a>
</p>

Single channel images of handwritten digits of size 28 x 28 pixels.


It can be obtained from [[5](#references)].

### Photon-Electron Electromagnetic Calorimeter (ECAL) Dataset

<p align="middle">
  <img src="https://raw.githubusercontent.com/eraraya-ricardo/qcnn-hep/main/assets/photon%20full.png" title="Photon" />
  <img src="https://raw.githubusercontent.com/eraraya-ricardo/qcnn-hep/main/assets/electron%20full.png" title="Electron" /> <br>
  <a>Averages of Photon (left) and Electron (right) image samples from the dataset.</a>
</p>
  
The dataset contains images from two types of particles: photons (0) and electrons (1) captured by the ECAL detector.
- Each pixel corresponds to a detector cell.
- The intensity of the pixel corresponds to how much energy is measured in that cell.
- In total, there are 498,000 samples, equally distributed between the two classes.
- The size of the images are 32x32.
  
If you are interested on using the datast for your study, contact me via [email](mailto:eraraya-ricardo@qlab.itb.ac.id) and I can connect you to the people at ML4Sci who have the dataset.

### Quark-Gluon Dataset

<p align="middle">
  <img src="https://raw.githubusercontent.com/eraraya-ricardo/qcnn-hep/main/assets/gluon-125-10k.png" title="Gluon" />
  <img src="https://raw.githubusercontent.com/eraraya-ricardo/qcnn-hep/main/assets/quark-125-10k.png" title="Quark" /> <br>
  <a>Averages of Gluon (left) and Quark (right) image samples of the track channel from the subdataset of 10k samples.</a>
</p>
  
<p align="middle">
  <img src="https://raw.githubusercontent.com/eraraya-ricardo/qcnn-hep/main/assets/gluon-40-10k.png" title="Gluon" />
  <img src="https://raw.githubusercontent.com/eraraya-ricardo/qcnn-hep/main/assets/quark-40-10k.png" title="Quark" /> <br>
  <a>Cropped to 40 x 40.</a>
</p>

The dataset contains images of simulated quark and gluon jets. The image has three channels, the first channel is the reconstructed tracks of the jet, the second channel is the images captured by the electromagnetic calorimeter (ECAL) detector, and the third channel is the images captured by the hadronic calorimeter (HCAL) detector.
  
- The images have a resolution of 125 x 125 pixels (for every channel).
- Since the original size of 125 x 125 pixels is too large for quantum computing simulation, we cropped the images into certain size. For now, we limit the current size to 40 x 40 pixels.
- In this study, we focus mostly on the tracks channel.
- You can check reference [[6](#references)] for more details of the dataset.

If you are interested on using the datast for your study, contact me via [email](mailto:eraraya-ricardo@qlab.itb.ac.id) and I can connect you to the people at ML4Sci who have the dataset.


