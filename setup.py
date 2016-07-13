from setuptools import setup

setup(
    name='ModularMailer-ClassifierNetwork',
    packages=['classifiernetwork'],
    version='0.0a1',
    description='Package for creating and using a classifier neural network',
    author='James Payne',
    author_email='jepayne1138@gmail.com',
    url='https://github.com/jepayne1138/ModularMailer-ClassifierNetwork',
    license='BSD-new',
    download_url='https://github.com/jepayne1138/ModularMailer-ClassifierNetwork/tarball/0.0a1',
    keywords='keras theano numpy scipy h5py',
    install_requires=['Keras', 'Theano', 'numpy', 'scipy', 'h5py'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: Implementation :: CPython',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Developers',
    ],
    entry_point={
        'console_scripts': [
            'network = classifiernetwork.console:main'
        ],
    }
)
