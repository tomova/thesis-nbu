# Multipole Moment Prediction with Symmetry-preserving Graph Neural Networks

This repository contains the code used in the master's thesis titled "Multipole Moment Prediction with Symmetry-preserving Graph Neural Networks". Here, you will find datasets and models utilized for predictions on molecular multipole moments.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Setup and Usage](#setup-and-usage)
    - [Requirements](#requirements)
    - [Dataset Preparation](#dataset-preparation)
    - [Models](#models)

## Directory Structure

* **dataset**: Contains scripts for downloading and processing the dataset.
* **models**: Houses the predictor classes for both dipole and quadrupole moments.

## Setup and Usage

### Requirements

Ensure you've installed all required Python packages using the provided `requirements.txt`:

`pip install -r requirements.txt`

### Dataset Preparation

To download and prepare the dataset, first navigate to the dataset directory, then execute the scripts:

`cd dataset`

`python download.py`,

`python prepare.py`

### Models
The  **models** directory contains the dipole and quadrupole predictor classes. Browse through to examine the implementations and to understand how to utilize them.