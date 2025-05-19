# NLP Project

## Installation

1. Clone the repository:

```bash
git clone git@github.com:andreafabbricatore/nlp-project.git
cd nlp-project
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Run files with:

```bash
poetry run python filename.py
```

4. To run notebooks:

```bash
poetry run python -m ipykernel install --user --name=nlp_project --display-name "nlp_project"
```

## Project Structure

```
nlp-project/
├── src/
│   └── nlp_project/
│       ├── data/           # Data processing and loading utilities
│       ├── models/         # Ensemble model architectures and definitions
│       ├── utils/          # Utility functions
│       ├── training/       # Training scripts and configurations
│       └── pii_decorator/  # Sample PII detection library
├── notebooks/              # Jupyter notebooks for development and analysis
│   ├── inference.ipynb     # Model inference examples
│   └── embedding_viz.ipynb # Embedding visualization
├── datasets/
│   └── datasets.txt        # Drive links to datasets
└── models/
    └── models.txt          # Drive links to models

## Authors

Group 2
