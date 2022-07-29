# <center>Attention LSTMs in Multimodal Models</center>

## Overview

This repository contains all models, experiments, and results from the paper _Attention LSTMs in Multimodal Models_. All models and experiments are implemented and executed with TensorFlow and Keras. Below is the organization structure:

- `AttentionBottleneckLSTM/` contains the overall utility file `att_bott_utils.py`, which has many helper functions as well as the model creation function for the Attention Bottleneck Mid Fusion Model.
	- `create_att_bottleneck_model()` creates the attention bottleneck mid fusion model
	- `load_sequential_data()` loads and transforms data into processible form for the model
	- `create_flow()` creates generators for model training and testing.
- `Conv2DAttentionLSTM/` contains files and helper functions for the Image Attention LSTM model. In `conv2d_mha_utils.py`, below are the important functions.
	- `create_conv_mha_lstm_model()` creates an Image Attention LSTM model
	- `load_sequential_data()` loads and transforms data specific to the model
	- `create_flow()` creates generators for training and testing
- `GraphAttentionLSTM/` contains files and helper functions for the Graph Attention LSTM model. In `mhga_utils.py`, below are the important functions.
	- `create_graph_attention_lstm_model()` creates a Graph Attention LSTM model
	- `load_sequential_data()` loads and transforms data specific to the model
	- `create_flow()` creates generators for training and testing
- `experiments/` are the colab notebooks used for experiments
- `Results_metric.xlsx` contains the organized results that are also shown in the paper
- `imports.py` is a file that contains all library imports needed for models. Due to some modules being session-based (Ex: TensorFlow), taking all imports from a single source makes sure only one session is created.

## Prediction results in 20 training epochs

### Attention Bottleneck Mid Fusion Model

![Alt Text](https://github.com/NuoWenLei/Attention-LSTMs-in-Multimodal-Models/blob/main/infection_maps_true_v_pred_attention.gif)

### LSTM Late Fusion Model

![Alt Text](https://github.com/NuoWenLei/Attention-LSTMs-in-Multimodal-Models/blob/main/infection_maps_true_v_pred_lstm.gif)
