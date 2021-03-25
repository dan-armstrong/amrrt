import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="amrrt",
    version="1.0",
    author="Dan Armstrong",
    author_email="danarmstrongg@gmail.com",
    description="Official release of AM-RRT*, the package also includes an implementation of RT-RRT*",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject", #CHANGE
    packages=setuptools.find_packages(),
    license='Apache License 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['cycler>=0.10.0',
                      'pytest>=5.4.3'
                      'decorator>=4.4.2',
                      'imageio>=2.8.0',
                      'kiwisolver>=1.1.0',
                      'matplotlib>=3.1.3',
                      'networkx>=2.4',
                      'numpy>=1.18.1',
                      'Pillow>=7.0.0',
                      'pygame>=1.9.6',
                      'pyparsing>=2.4.6',
                      'python-dateutil>=2.8.1',
                      'PyWavelets>=1.1.1',
                      'scikit-image>=0.16.2',
                      'scipy>=1.4.1',
                      'six>=1.14.0',
                      'vptree>=1.2']
)
