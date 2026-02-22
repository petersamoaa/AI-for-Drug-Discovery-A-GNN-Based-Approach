# Module 1: Molecular Fundamentals & Graph Representations

## 1. Introduction: The Graph Nature of Molecules
In chemistry and molecular sciences, a prominent problem has been representing molecules in a general, application-agnostic way and inferring possible interfaces between molecules, such as proteins. The molecules representation is essentially graph structures:
- **Nodes ($V$):** Represent individual Atoms (Carbon, Nitrogen, Oxygen, etc.).
- **Edges ($E$):** Represent Atomic Bonds.

![Molecular Graph Representation](../images/molecular_graph_mapping.png)

*Figure 1.1: Visualizing how a caffeine molecule translates into a mathematical graph object.*
## 2. GNNs vs. Traditional "Fingerprints"
Traditionally, molecular properties were determined using "fingerprint" methods. These required domain experts to create manual features based on the presence or absence of specific sub-structures. 

**Why GNNs are better:**
GNNs learn **data-driven features**. They can group molecules in unexpected ways and propose new synthesis routes. This is critical for predicting:
- **Toxicity:** Is the chemical safe for human use?
- **Efficacy:** Does it have the intended biological effect on disease progression?

## 3. Molecular Data Formats: SMILES & ZINC
To process molecules in Python, we use the **SMILES** (Simplified Molecular Input Line Entry System) format. As described in **Section 5.4.1 (Page 178)**, SMILES represents molecular graphs in ASCII format.

**The ZINC Dataset:**
Our training usually involves the ZINC dataset (~250,000 molecules), which includes:
1. **logP:** The water-octanol partition coefficient (measures solubility).
2. **SAS:** Synthetic Accessibility Score (how hard is it to make?).
3. **QED:** Quantitative Estimate of Druglikeness (the "gold standard" for how much a molecule looks like a potential drug).

## 4. The GNN Workflow for Drug Discovery
As illustrated in **Figure 1.12 (Page 19)**, the workflow consists of an **Encoder** that transforms the molecular graph into a **Latent Representation** (a vector of numbers). This representation is then used to predict properties or "dream" of new molecules via a **Decoder**.

---

## 💻 Implementation: Converting SMILES to Graphs
The following implementation uses **RDKit** and **PyTorch Geometric** to convert a SMILES string into a GNN-ready Data object.

### Listing 5.17: SMILES to Graph Function (Page 179)
```python
from torch_geometric.data import Data
import torch
from rdkit import Chem

def smiles_to_graph(smiles, qed):
    # Load molecule from SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # Extract Edges and Bond Features
    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        # Encoding bond types (Single, Double, etc.)
        bond_type = bond.GetBondTypeAsDouble()
        bond_feature = [1 if i == bond_type else 0 for i in range(4)]
        edge_features.append(bond_feature)
        
    # Convert to Tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], 
                     dtype=torch.float).view(-1, 1)
    
    return Data(x=x, edge_index=edge_index, qed=torch.tensor([qed]))
