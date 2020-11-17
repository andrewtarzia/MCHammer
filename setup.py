import setuptools

with open("readme.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MCHammer",
    version="0.0.1",
    author="Andrew Tarzia",
    author_email="andrew.tarzia@gmail.com",
    description="Contains MC algorithm for optimising molecules.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andrewtarzia/MCHammer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
