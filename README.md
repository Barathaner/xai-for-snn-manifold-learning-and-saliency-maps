# Explainable AI: Visualization of Neural Activity Dynamics of Spiking Neural Networks via Manifold Learning and Saliency Maps
Spiking Neural Networks (SNNs) represent the third generation of neural networks. Inspired by biological neurons, they are inherently more complex but hold great promise for energy-efficient applications and neuroscience research. This project focuses on the explainability of SNNs (XAI).

The central hypothesis is: Manifolds reveal structural insights into SNNs.
To investigate this, a full machine learning pipeline is implemented using snntorch.

The project applies manifold learning techniques (SPUD) to visualize the dynamic neural activity within SNNs. In addition, saliency map methods (Spiking-LIME) are used to assess the relevance of input features.

The work includes both a theoretical analysis and experimental evaluation. Expectations are compared with empirical results, revealing the types of insights that visualizations can provide. The final outcome is a comprehensive overview of visualization techniques for SNNs.

## Techniques used in this repository

- PCA
- t-SNE
- UMAP
- LIME
- SPUD
- Saliency Maps

## Repository Structure:

- **data/**  
  Enthält Datensätze und Preprocessing-Skripte für die Experimente mit SNNs.
- **experiments/**  
  some running scripts of the full pipeline and to load weights
- **model_export/**  
  export of the trained SNN as NIR and as pth
- **models/**
definition of the architecture of the snns
- **utils/**  
  Hilfsfunktionen und allgemeine Tools, die in mehreren Modulen verwendet werden.
- **training/**
- definition of the training algorithm and callbacks or evaluations
