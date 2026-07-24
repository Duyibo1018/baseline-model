Neural Network Models

This repository contains three neural-network models for wind-farm performance prediction: WPGNN/GAT, CNN, and MLP.

1. Data Input

GAT

The graph-based model uses turbine-level, edge-level, and global features.

Global input (x_globals): 3 features

Node input (x_nodes): 3 features per turbine

Edge input (x_edges): 2 features per edge

Target: graph-level output from f_globals

The graph connectivity is defined by the sender and receiver indices in the dataset.

CNN

The CNN uses a four-channel rasterized wind-farm layout:

Input shape: 4 × 64 × 64

The four channels are:

Turbine occupancy map

Global feature 1

Global feature 2

Global feature 3

MLP

For the variable-turbine case, the MLP uses a fixed-length input vector:

80 × 2 turbine coordinates
+ 80 turbine-existence mask values
+ wind speed
+ wind direction
+ turbulence intensity
= 243 features

2. Model Architectures

WPGNN / GAT

Graph input
    ↓
Graph attention / message passing layers
    ↓
Graph-level feature aggregation
    ↓
Regression output

Main training hyperparameters:

Learning rate: 1e-4

Learning-rate decay: 0.999

Batch size: 128

Epochs: 500

Print interval: 100

Save interval: 50

Pretrained model: wpgnn.h5

CNN

4 × 64 × 64
    ↓
Conv: 4 → 32
    ↓
Conv: 32 → 64
    ↓
Conv: 64 → 128
    ↓
Adaptive Average Pooling
    ↓
MLP: 2048 → 64 → 32 → 1

Main hyperparameters:

Kernel size: 3 × 3

Activation: SiLU

Dropout: 0.10

Learning rate: 3e-4

Batch size: 64

MLP

243
 ↓
512
 ↓
256
 ↓
128
 ↓
64
 ↓
1

Main hyperparameters:

Activation: ReLU

Dropout: 0.15

Learning rate: 3e-4

Batch size: 16

3. Output

All models output a single scalar representing the predicted wind-farm-level target.
