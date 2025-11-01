from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Read and return a list of requirements from the given file."""
    with open(file_path) as file_obj:
        requirements = [line.strip() for line in file_obj if line.strip()]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='Product_Return_Predictions_Final',
    version='1.0.0',
    author='Naga Sathvik ',
    author_email='sathvikvedantham@gmail.com',
    description='A machine learning project to predict product returns in e-commerce.',
    long_description=open('README.md').read() if open else '',
    long_description_content_type='text/markdown',
    url='https://github.com/YourGitHubUsername/Product_Return_Predictions_Final',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
