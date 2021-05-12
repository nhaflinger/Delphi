import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="delphi", 
    version="0.0.1",
    author="Doug Creel",
    author_email="mrsunshine001@gmail.com",
    description="Telemetry analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https:https://dev1.masten.aero/Utilities/Delphi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
