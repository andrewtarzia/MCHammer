import setuptools

setuptools.setup(
    name="MCHammer",
    version="1.0.0",
    author="Andrew Tarzia",
    author_email="andrew.tarzia@gmail.com",
    description="Contains MC algorithm for optimising molecules.",
    url="https://github.com/andrewtarzia/MCHammer",
    packages=setuptools.find_packages(),
    install_requires=(
        'scipy',
        'matplotlib',
        'networkx',
        'numpy',
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
