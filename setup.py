from setuptools import setup, find_packages
from codecs import open
from os import path

# directory containing this file
HERE = path.abspath(path.dirname(__file__))

# long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# call to setup() does all the work
setup(
    name="FISH-analysis",
    version="0.0.1",
    description="FISH image analysis library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sreevatsa03/FISH-Image-Analysis",
    author="Sreevatsa Nukala, Kate Hudson, Antonio Villaneuva, Soumili Dey, Joseph Entner",
    author_email="sreevatsa.nukala@gmail.com",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing"
    ],
    packages=["FISH_analysis"],
    include_package_data=True,
    install_requires=["numpy",
                    "cellpose", 
                    "Pillow", 
                    "matplotlib", 
                    "os", 
                    "opencv-python", 
                    "scikit-learn"]
)