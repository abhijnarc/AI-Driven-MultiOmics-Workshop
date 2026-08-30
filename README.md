# AI-Driven Multi-Omics Integration

## Conference

**Genomics India Conference 2026**

### Workshop
**AI-Driven Multi-Omics Integration: Principles, Models and Translation**

---

## Repository

This repository contains the Jupyter notebooks and frozen datasets used for the hands-on demonstrations in the workshop.

The notebooks demonstrate how different statistical and AI-based approaches can be used to learn representations and integrate heterogeneous multi-omics data.

Participants can clone or download this repository and run the notebooks directly using the supplied frozen datasets.

---

## Notebooks

### 1. Disease Subtyping with VAE
**Variational Autoencoder (VAE)**

Introduces nonlinear representation learning for multi-omics data and demonstrates how a compact latent representation can capture molecular structure across patients.

### 2. Shared Patterns with CCA
**Canonical Correlation Analysis (CCA) + Neural Shared Projection**

Demonstrates how shared patterns between different omics layers can be identified using a classical statistical method and compared with a nonlinear neural approach.

### 3. Multi-Omics Latent Factors
**MOFA + Attention**

Introduces multi-view latent-factor learning and attention-based integration to explore shared and modality-specific patterns across multiple omics layers.

### 4. Patient Similarity and Graph Learning
**Similarity Kernel + Graph Attention Network (GAT)**

Represents patients as nodes in a molecular similarity graph and demonstrates how graph-based deep learning can learn representations from relationships between patients.

### 5. Molecular Subtype Prediction
**Multimodal Transformer**

Demonstrates self-attention for integrating multiple omics representations and predicting molecular breast-cancer subtypes.

---

## Plant Multi-Omics Demonstration

A supplementary demonstration uses real **Arabidopsis thaliana WallOmics** data containing matched transcriptomics, proteomics and metabolomics measurements.

The demonstration uses a **MOGONET-style graph-based architecture** to show how AI-driven multi-omics integration can also be applied to plant and agricultural research.

---

## Data

The repository contains frozen, processed workshop inputs. Participants do not need to download or preprocess the original raw datasets.

Simply clone/download the repository and run the notebooks using the supplied data.
