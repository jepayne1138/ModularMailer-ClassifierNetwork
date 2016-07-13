"""Handle the console interface for the ClassifierNetwork package.

This package, while intended to be used with the ModularMailer project, can
be installed and used a standalone application, which this module handles
the interface for. The intent of of this design is so that this package can
be installed by itself without the ModularMailer project as a dependency on
a machine that is optimized for training the network (Linux box w/GPU).
"""
import argparse
import sys
import classifiernetwork.defaults as defaults


SUPPORTED_OBJECTIVES = (
    'mean_squared_error',
    'mse',
    'mean_absolute_error',
    'mae',
    'mean_absolute_percentage_error',
    'mape',
    'mean_squared_logarithmic_error',
    'msle',
    'squared_hinge',
    'hinge',
    'binary_crossentropy',
    'categorical_crossentropy',
    'sparse_categorical_crossentropy',
    'kullback_leibler_divergence',
    'kld',
    'poisson',
    'cosine_proximity',
)

SUPPORTED_ACTIVATIONS = (
    'softmax',
    'softplus',
    'softsign',
    'relu',
    'tanh',
    'sigmoid',
    'hard_sigmoid',
    'linear',
)


def _build_training_subparser(train_parser):
    """Create the options for the 'train' subparser"""
    train_parser.add_argument(
        'input_vectors', type=str,
        help='Path to the numpy array of input vectors (.npy file).'
    )
    train_parser.add_argument(
        'output_vectors', type=str,
        help='path to the numpy array of output vectors (.npy file)'
    )
    train_parser.add_argument(
        'save_name', type=str, help='Save trained network file name.'
    )
    train_parser.add_argument(
        '-o', '--output-directory', type=str,
        help='Directory for output file. Defaults to input_vectors location.'
    )

    # Network compilation option
    compile_group = train_parser.add_argument_group(
        title='Compilation options',
        description='Options for the structure of the network.'
    )
    compile_group.add_argument(
        '-i', '--hidden-size', type=int,
        help='Size of the hidden layer. Defaults to geometric_mean(in, out).'
    )
    compile_group.add_argument(
        '-a', '--activation', type=str,
        default=defaults.ACTIVATION, choices=SUPPORTED_ACTIVATIONS,
        help='Activation function for the hidden layer (see Keras docs).'
    )
    compile_group.add_argument(
        '-p', '--dropout', type=float, default=defaults.DROPOUT,
        help='Fraction of the input units to drop.'
    )
    compile_group.add_argument(
        '-l', '--loss', type=str,
        default=defaults.LOSS, choices=SUPPORTED_OBJECTIVES,
        help='The string identifier of an optimizer (see Keras docs).'
    )

    # Options for the stochastic gradient descent optimizer
    sgd_group = train_parser.add_argument_group(
        title='Stochastic Gradient Descent optimizer (SGD) options',
        description='The network is trained using a SGD optimizer.'
    )
    sgd_group.add_argument(
        '-r', '--learning-rate', type=float, default=defaults.LEARNING_RATE,
        help='Learning rate.'
    )
    sgd_group.add_argument(
        '-m', '--momentum', type=float, default=defaults.MOMENTUM,
        help='Number of epochs to train the network.'
    )
    sgd_group.add_argument(
        '-d', '--decay', type=float, default=defaults.DECAY,
        help='Learning rate decay over each update.'
    )
    sgd_group.add_argument(
        '-n', '--nesterov', action='store_true',
        help='Apply Nesterov momentum to the SGD optimizer.'
    )

    # Options for training the model
    train_group = train_parser.add_argument_group(
        title='Training options',
        description='Options for how the network is to be trained.'
    )
    train_group.add_argument(
        '-e', '--epochs', type=int, default=defaults.EPOCH,
        help='The number of epochs to train the model.'
    )
    train_group.add_argument(
        '-s', '--validation-split', type=float,
        help='Fraction of the data to use as held-out validation data.'
    )
    train_group.add_argument(
        '--v', '--verbose', type=int,
        default=defaults.VERBOSE, choices=(0, 1, 2),
        help='0 for no logging, 1 for progress bar, 2 for line per epoch.'
    )
    train_group.add_argument(
        '-b', '--batch-size', type=int,
        help='Number of samples per gradient update.'
    )


def argument_parser(args):
    parser = argparse.ArgumentParser(
        description='Trains neural networks from labeled input data.'
    )

    # Create subparser
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    # Parse 'train' command
    train_parser = subparsers.add_parser(
        'train', help='Train a neural network from the given input.'
    )
    _build_training_subparser(train_parser)

    # Return parsed arguments
    return parser.parse_args(args)


def main():
    """Entry point for the console script usage of this package.

    Returns:
        int: Error return code.
    """
    args = argument_parser(sys.argv[1:])

    return 0
