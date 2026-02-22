# Module 1: Molecular Fundamentals & Graph Representations

## 1. Introduction: The Graph Nature of Molecules
In chemistry and molecular sciences, a prominent problem has been representing molecules in a general, application-agnostic way and inferring possible interfaces between molecules, such as proteins. The molecules' representation is essentially a graph structure:
- **Nodes ($V$):** Represent individual Atoms (Carbon, Nitrogen, Oxygen, etc.).
- **Edges ($E$):** Represent Atomic Bonds.

![Molecular Graph Representation](../images/molecular_graph_mapping.png)

*Figure 1.1: Visualising how a caffeine molecule translates into a mathematical graph object.*
## 2. GNNs vs. Traditional "Fingerprints"
Traditionally, molecular properties were determined using "fingerprint" methods. These required domain experts to manually create features based on the presence or absence of specific substructures. 

**Why GNNs are better:**
GNNs learn **data-driven features**. They can group molecules in unexpected ways and propose new synthesis routes. This is critical for predicting:
- **Toxicity:** Is the chemical safe for human use?
- **Efficacy:** Does it have the intended biological effect on disease progression?

Drug discovery, especially for GNNs, can be understood as a **graph prediction problem**. Graph prediction tasks require learning and predicting properties of the entire graph. In drug discovery, the aim is to predict properties such as toxicity or treatment effectiveness (discriminative) or to suggest entirely new compounds to be synthesised and tested (generative). To suggest these new graphs, drug discovery methods often combine GNNs with other generative models such as variational graph autoencoders (VGAEs), as shown in Figure 1.2. 

![Molecular Graph Representation](../images/Molecular_graph_Pipeline.png)

*Figure 1.2: A GNN system used to predict new molecules. The workflow here starts on the left with a graph representation of a molecule. In the middle parts of the figure, this graph representation is transformed via a GNN into a latent representation. The latent representation is then transformed back to the molecule to ensure that the latent space can be decoded (right).*

Moreover, the representation of molecules captures the inherent sparsity of molecular structures, where most atoms form only a few bonds, and large portions of the molecule may be distant from each other in the graph. Traditional machine learning methods often struggle to predict properties of new molecules due to this sparsity, as they don’t account for the full structural context. GNNs overcome these challenges by capturing both local atomic environments and global molecular structures. GNNs learn hierarchical features from fine-grained atomic interactions to broader molecular properties,
and their ability to remain invariant to the ordering of atoms ensures consistent predictions. By leveraging molecular graph structure, GNNs make accurate predictions from sparse, connected data, thereby accelerating the drug discovery process.

## 3. Molecular Data Formats: SMILES & ZINC
To process molecules in Python, we use the **SMILES** (Simplified Molecular Input Line Entry System) format. SMILES represents molecular graphs in ASCII format.

**The ZINC Dataset:**
Our training usually involves the ZINC dataset (~250,000 molecules), which includes:
1. **logP:** The water-octanol partition coefficient (measures solubility).
2. **SAS:** Synthetic Accessibility Score (how hard is it to make?).
3. **QED:** Quantitative Estimate of Druglikeness (the "gold standard" for how much a molecule looks like a potential drug).

## 4. The GNN Workflow for Drug Discovery
As illustrated in **Figure 1.2**, the workflow consists of an **Encoder** that transforms the molecular graph into a **Latent Representation** (a vector of numbers). This representation is then used to predict properties or "dream" of new molecules via a **Decoder**.

---

## 💻 Implementation: Converting SMILES to Graphs
The following implementation uses **RDKit** and **PyTorch Geometric** to convert a SMILES string into a GNN-ready Data object.

To make the ZINK dataset usable by GNN models, we need to convert it into a suitable graph structure. Here, we’re going to use PyG to define our model and run deep learning routines. Therefore, we first download the data and then convert the dataset into graph objects using NetworkX.

### Create a molecular graph dataset
```python
import requests
import pandas as pd

def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'wb') as f:
    f.write(response.content)
    
url = "https://raw.githubusercontent.com/
aspuru-guzikgroup/chemical_vae/master/models/
zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
filename = "250k_rndm_zinc_drugs_clean_3.csv"
download_file(url, filename)
df = pd.read_csv(filename)
df["smiles"] = df["smiles"].apply(lambda s: s.replace("\n", ""))
```

We downloaded the ZINK dataset via cdode above, which generates the following output:
smiles logP qed SAS
0 CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1
5.05060 0.702012 2.084095
1 C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1
3.11370 0.928975 3.432004
2 N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)...
4.96778 0.599682 2.470633
3 CCOC(=O)[C@@H]1CCCN(C(=O)c2nc
(-c3ccc(C)cc3)n3c...
4.00022 0.690944 2.822753
4 N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])
[C@H](C#... 3.60956 0.789027 4.035182

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
