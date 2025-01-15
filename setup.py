from setuptools import setup, find_packages

setup(
    name="anomalie_eth",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "web3",
        "python-dotenv",
        "streamlit",
        "plotly",
        "mlflow",
        "requests",
        "tqdm",
        "pytest",
        "pytest-cov"
    ],
    python_requires=">=3.9",
) 