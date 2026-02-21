# Module 1: Molecular Fundamentals

Before we can train a model to find a cure for a disease, we have to teach it how to "see" a molecule.

## 1. The Problem with SMILES
In traditional chemistry, we use **SMILES** strings (Simplified Molecular Input Line Entry System). 
Example of Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`

**The Issue:** A string is a sequence. If you change one letter, the whole meaning changes. However, molecules are **graphs**. They are spatial and connected.

## 2. Molecules as Graphs
A GNN treats a molecule as a mathematical graph $G = (V, E)$:
* **Nodes (V):** The Atoms (Carbon, Nitrogen, Oxygen).
* **Edges (E):** The Chemical Bonds (Single, Double, Triple).



## 3. Why GNNs?
Standard Neural Networks are not **Permutation Invariant**. If you swap the order of the atoms in a list, a standard network gets confused. A GNN doesn't care about the order; it only cares about the **connections**.

## 4. Key Metrics: QED & LogP
When evaluating a molecule, we look at:
* **QED (Quantitative Estimate of Druglikeness):** How much does this "look" like a real drug?
* **LogP (Solubility):** Can this drug actually dissolve in the human body?

---

## 💻 Hands-on: Turning SMILES into Graphs
In the accompanying notebook, we use **RDKit** to perform this conversion.

### [Go to Jupyter Notebook Implementation](../notebooks/01_molecular_viz.ipynb)
