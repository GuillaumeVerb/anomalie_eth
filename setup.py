from setuptools import setup, find_packages

setup(
    name="eth_anomaly_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas==2.0.0',
        'numpy==1.23.5',
        'scikit-learn==1.2.0',
        'web3==6.11.1',
        'streamlit==1.29.0',
        'python-dotenv==1.0.0',
        'pytest==7.3.1',
        'jupyter==1.0.0',
        'tqdm==4.65.0',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'plotly==5.18.0',
        'pyarrow==14.0.1',
        'mlflow==2.9.2'
    ]
) 