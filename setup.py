from setuptools import setup, find_packages

setup(
    name="rough_path_forecaster",
    version="1.0.0",
    author="P2SAMAPA",
    description="Signature kernel + Log-ODE ETF forecasting engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "gpytorch>=1.10",
        "huggingface-hub>=0.20.0",
        "pyarrow>=14.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "statsmodels>=0.14.0",
        "torchdiffeq>=0.2.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
