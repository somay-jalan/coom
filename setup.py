from setuptools import setup, find_packages

setup(
    name='coom',
    version='0.1.0',
    description='Large Scale LLM Pretraining Codebase',
    author='Soket AI',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.10',
    install_requires=[
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