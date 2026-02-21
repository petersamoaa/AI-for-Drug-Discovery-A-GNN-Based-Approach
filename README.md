# AI-for-Drug-Discovery-A-GNN-Based-Approach
🧬 AI for Drug Discovery: A GNN-Based Approach
This repository is a hands-on guide to applying Graph Neural Networks (GNNs) to computational biology and chemistry. While standard Neural Networks treat data as grids (images) or sequences (text), GNNs allow us to model the natural world as it is: a web of interactions.

🔍 Featured Topic: Protein-Protein Interaction (PPI)
What is it? > Most biological processes are governed by "molecular handshakes" between proteins. If we can predict which proteins interact, we can identify drug targets for diseases like cancer or Alzheimer's.

The Model: GraphSAGE
In this repository, we implement GraphSAGE (SAmple and aggreGatE). We use this specifically for PPI because:

Inductive Nature: It can predict interactions for proteins it has never seen during training.

Scalability: It uses neighborhood sampling to handle massive protein networks without crashing your GPU memory.
