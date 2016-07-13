ModularMailer-ClassifierNetwork
========================

Description
-----------
This package manages a neural network specifically for the ModularMailer project. It additionally contains a console interface (and no dependency on the ModularMailer main application itself) so that it can be installed separately on any computer to train a neural network from saved numpy array files. In this way, even if the ModularMailer application is installed on a computer that is not optimized from training the network, this package can be used independently to train the necessary network on a better machine (Linux box with GPU). It therefore exposes an expansive set of network creation and training options using the 'network train' command.

Installation
------------
Install from PyPI using::

    pip install ModularMailer-ClassifierNetwork

Installation requires (and will attempt to install) the following dependencies:

* numpy
* scipy
* Theano
* Keras
* h5py
