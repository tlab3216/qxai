"""
Setup script for Q-XAI: Interpretable Complex-Valued Transformers for Acoustic Scene Classification
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f.readlines() 
                          if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name="q-xai",
    version="1.0.0",
    description="Interpretable Complex-Valued Transformers for Acoustic Scene Classification",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Anonymous AAAI 2026 Submission",
    author_email="anonymous@submission.com",
    url="https://github.com/anonymous/q-xai",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "experiments": [
            "hydra-core>=1.1.0",
            "wandb>=0.12.0",
            "mlflow>=1.20.0",
        ]
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "q-xai-train=experiments.train_q_xai:main",
            "q-xai-evaluate=experiments.evaluate_model:main",
            "q-xai-reproduce=experiments.reproduce_results:main",
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Signal Processing",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    
    # Keywords
    keywords=[
        "acoustic scene classification",
        "complex-valued neural networks", 
        "transformer",
        "explainable AI",
        "uncertainty quantification",
        "conformal prediction",
        "quantum-inspired",
        "interpretability"
    ],
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/anonymous/q-xai/issues",
        "Source": "https://github.com/anonymous/q-xai",
        "Documentation": "https://q-xai.readthedocs.io/",
    },
    
    # License
    license="MIT",
    
    # Zip safety
    zip_safe=False,
)