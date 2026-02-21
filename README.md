# 🧬 AI for Drug Discovery: A GNN Handbook

Welcome! This repository provides a curated learning path for applying **Graph Neural Networks (GNNs)** to drug discovery and molecular science. 

This project bridges the gap between theoretical chemistry and deep learning by implementing concepts from *Graph Neural Networks in Action* (Manning) and *Hands-On GNNs* (Packt).

---

## 🗺️ Learning Path

Explore the modules below to understand how AI is revolutionising the discovery of new medicines.

### [Module 1: Molecular Fundamentals](./chapters/01_fundamentals.md)
*Learn why molecules are graphs and how to represent them in Python using RDKit.*

### [Module 2: Protein-Protein Interaction (PPI)](./chapters/02_ppi.md)
*Understanding large-scale biological interactions using the GraphSAGE architecture.*

### [Module 3: Generative Discovery](./chapters/03_generative.md)
*Using Generative Adversarial Networks (MolGAN) to "dream up" new drug candidates.*

---

## 🛠️ Tech Stack
* **Language:** Python
* **Cheminformatics:** RDKit, DeepChem
* **Deep Learning:** PyTorch Geometric (PyG)
* **Datasets:** ZINC, Tox21, PPI

---

## 🚀 Getting Started
To run the code examples, clone this repo and install the requirements:
```bash
pip install torch-geometric rdkit deepchem
