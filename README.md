# DeepSUM-DWI
Image registration and super-resolution network for diffusion-weighted imaging (DWI) MRI, based on the DeepSUM model (https://github.com/diegovalsesia/deepsum).

DeepSUM is a Multi Image Super-Resolution (MISR) deep neural network that exploits both spatial and temporal correlations to recover a single high resolution image from multiple unregistered low resolution images. In this implementation it is updated to work with TensorFlow2 and modified for use with repeated diffusion-weighted image inputs.

This project was funded by UNAM-PAPIIT grant: TA101224.

## Setup
DeepSUM-DWI uses Tensorflow 2.20. The full requirements can be found in the requirements.txt file. It is recommended to follow the official tensorflow instructions for installation: https://www.tensorflow.org/install/pip 

## Usage
Three notebooks are provided, similar to in the original DeepSUM implementation, which demonstrates the usage of the code: one for the dataset creation and preparation, one for the model training, and one for model prediction.

## Authors & Contacts
The original DeepSUM code is based on work by team SuperPip from the Image Processing and Learning group of Politecnico di Torino: Andrea Bordone Molini (andrea.bordone AT polito.it), Diego Valsesia (diego.valsesia AT polito.it), Giulia Fracastoro (giulia.fracastoro AT polito.it), Enrico Magli (enrico.magli AT polito.it). The original team were not involved in this current implementation.

The DWI extension was implemented by a group of the National MRI Laboratory of the Universidad Nacional Autónoma de México, Mexico: Merlin Fair (merlin.fair AT unam.mx), Carolina Daniells, Luis Concha and Guadalupe Garcia.

## Acknowledgements
This project was funded by UNAM-PAPIIT grant: TA101224.

