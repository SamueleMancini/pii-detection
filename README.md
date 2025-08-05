# NLP Project

This work presents a robust and lightweight framework for detecting PII in real-world LLM prompts. By fine-tuning *ALBERT* and *DistilBERT* models and combining them through word-level ensembling, we improve detection performance and generalization.

Our main contribution is the introduction of adversarial training using evolutionary perturbations, which significantly boosts robustness to noisy or obfuscated inputs. The resulting models achieve over 94\% recall on adversarial test data, showing how iterative implementation of these strengthening methods could be the right pipeline towards more secure systems.

Embedding visualizations confirm that adversarial fine-tuning sharpens token representations, reinforcing the model's ability to distinguish sensitive entities under context shift.

Overall, our results suggest that transformer-based PII detectors—especially when adversarially augmented—offer a scalable and practical solution for privacy protection in LLM deployments.

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
