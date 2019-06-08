# ECE285_NST
Description
===========
This is project Style Transfer developed by team NST composed of Yonghong Liu, Xiaotao Guo

Code organization
=================
- demo.ipynb -- Run a demo of our code (reproduces Figure 11 - 12 of our report)
- real-time_style_transfer.ipynb -- Run the training of real-time style transfer (as described in Section 2)
- Experiment_classic_style_transfer.ipynb -- Run the experiment of classic style transfer (reproduces Figure 2 - 5 of our report)
- Experiment_realtime_style_transfer.ipynb -- Run the experiment of real-time style transfer (reproduces Figure 6 - 9 of our report)
- Cross experiments.ipynb -- Run the experiment of comparison between two versions of style transfer (reproduces Figure 10 of our report)
- original_NST.py -- Module for implementing classic style transfer (as described in Section 2) 
- train_transformer.py -- Module for implementing real-time style transfer (as described in Section 2)
- nntools_RTST.py -- Module for implementing checking point for training real-time style transfer （edited from nntools.py provided in pervious assignments)
- models/* -- trained real-time style transfer models
- logs/* -- the log for the traning of real-time style transfer models 
- images/content/* -- content image used for testing style transfer models 
- images/style/* -- style image used for training and testing style transfer models
