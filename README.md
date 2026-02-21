# 🧬 AI for Drug Discovery: A GNN-Based Handbook

This repository serves as a technical guide and portfolio for applying **Graph Neural Networks (GNNs)** to molecular science and drug discovery. 

Rather than treating molecules as simple strings or fixed vectors, we leverage their natural structure as mathematical graphs. This handbook follows a learning path inspired by the methodologies in *Graph Neural Networks in Action* (Manning) and *Hands-On GNNs* (Packt).

---

## 🗺️ Learning Path

### [Module 1: Molecular Fundamentals & Representations](./chapters/01_molecular_fundamentals.md)

An exploration of how molecules are discretised into graph objects, the limitations of traditional fingerprints, and the metrics used to evaluate druglikeness.

### [Module 2: Protein-Protein Interaction (PPI)](./chapters/02_ppi.md)

Applying the GraphSAGE architecture to large-scale biological networks to predict how proteins interact.

### [Module 3: Generative Molecular Design](./chapters/03_generative_discovery.md)

Using Variational Graph Autoencoders (VGAEs) to optimise latent spaces for de novo drug discovery.

---

## 🛠️ Requirements
To run the implementations in this handbook, you will need:
- `rdkit`: The industry standard for cheminformatics.
- `torch-geometric`: For building GNN layers.
- `pandas/numpy`: For data manipulation.

```bash
pip install torch-geometric rdkit pandas numpy
