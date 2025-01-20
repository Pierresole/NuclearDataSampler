from setuptools import setup, find_packages

setup(
    name="NuclearDataSampler",  # Replace with your project name
    version="0.0.0",  # Initial version
    description="Sample Nuclear Data Files",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="SOLE Pierre",
    author_email="pierre.sole@asnr.fr",
    url="https://github.com/Pierresole/NuclearDataSampler.git",  
    packages=find_packages(where="sources"),  # Automatically find and include all packages
    package_dir={"": "sources"},  # Root for all packages
    install_requires=[
        "h5py>=3.12.1",
        "numpy>=2.2.2",
        "pyDOE3>=1.0.4",
        "scipy>=1.15.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
)


