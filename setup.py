from setuptools import setup, find_packages

setup(
    name='coom',
    version='0.1.0',
    description='Large Scale LLM Pretraining Codebase',
    author='Soket AI',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    python_requires='>=3.10',
    install_requires=[
        'pytorch-lightning',
        'omegaconf',
        'hydra-core',
        'nemo_toolkit[all]',  
    ],
    extras_require={
        'dev': [
            'pytest',
            'black',
            'isort',
            'flake8',
        ]
    },
) 