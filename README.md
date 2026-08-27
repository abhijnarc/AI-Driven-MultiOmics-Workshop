# AI-Driven Multi-Omics Analysis Workshop

Hands-on workshop on **Advanced AI/ML in Genomics**, using TCGA-BRCA
multi-omics data to demonstrate how modern AI models can learn,
integrate, and interpret heterogeneous molecular representations.

## Workshop overview

This workshop treats multi-omics integration as a
**representation-learning problem**: RNA-seq, miRNA, and DNA methylation
measurements are transformed into increasingly structured latent
representations while preserving biologically meaningful relationships.

The five notebooks form a progressive pipeline:

  -----------------------------------------------------------------------
  Notebook          Use case          Core method       Advanced AI
                                                        component
  ----------------- ----------------- ----------------- -----------------
  01                Disease subtyping Variational       Nonlinear
                                      Autoencoder (VAE) probabilistic
                                                        representation
                                                        learning

  02                Shared molecular  CCA + Neural      Nonlinear neural
                    patterns          shared projection representation
                                                        alignment

  03                Latent factor     MOFA + attention  Learned
                    discovery         integration       cross-omics
                                                        attention

  04                Patient           Integrated        Graph attention
                    similarity        kernel + GAT      and message
                                                        passing

  05                Subtype           Multi-omics       Self-attention
                    identification    Transformer       across omics
                                                        tokens
  -----------------------------------------------------------------------

The progression moves from nonlinear latent representations to neural
alignment, attention-based integration, graph learning, and finally
Transformer-based cross-omics modelling.

------------------------------------------------------------------------

## Scientific objectives

The workshop demonstrates how AI can be used to:

1.  Learn compact representations from heterogeneous molecular data.
2.  Identify shared molecular structure across omics layers.
3.  Discover latent factors explaining multi-omics variation.
4.  Model patient-to-patient relationships using graphs.
5.  Integrate multiple omics views using attention and Transformers.
6.  Evaluate representations using biologically meaningful external
    labels.
7.  Compare advanced AI approaches against simpler baselines rather than
    assuming that a more complex model is automatically better.

**Important:** PAM50 labels are used as an external/post-hoc biological
reference where applicable. They are not used to construct unsupervised
representations or train the unsupervised integration stages.

------------------------------------------------------------------------

## Dataset

The workshop uses a frozen **TCGA-BRCA cohort of 617 patients**.

Primary molecular modalities:

-   RNA-seq
-   miRNA
-   DNA methylation

The processed data are stored under:

``` text
data/processed/
```

Raw inputs, where required, are kept separately from processed workshop
outputs.

### Main processed data

Representative processed files include:

``` text
TCGA_BRCA_RNA_selected.tsv
TCGA_BRCA_miRNA_selected.tsv
TCGA_BRCA_methylation_selected.tsv
TCGA_BRCA_cohort_metadata.tsv
TCGA_BRCA_cohort_freeze.json
```

Notebook-specific outputs are written under:

``` text
data/processed/
data/processed/figures/
```

------------------------------------------------------------------------

# Notebook 1 --- Disease Subtyping

### Method

**Variational Autoencoder (VAE)** with a deterministic Autoencoder
comparison.

### Architecture

The multi-omics input contains:

-   1,000 RNA features
-   300 miRNA features
-   1,000 methylation features

Total:

``` text
2,300 features
```

The VAE uses a nonlinear architecture:

``` text
2300 → 256 → 64 → latent(10)
```

with μ/logσ² heads and a mirrored decoder.

The model contains approximately **1.22 million parameters** and learns
a 10-dimensional probabilistic representation.

### Results

Validation reconstruction MSE:

  Model     Reconstruction MSE
  ------- --------------------
  VAE                  0.59280
  AE                   0.58143

The deterministic AE reconstructed marginally better, with approximately
**1.92% lower validation MSE**.

Post-hoc PAM50 agreement:

  Model        ARI      NMI
  ------- -------- --------
  VAE       0.2396   0.3860
  AE        0.2741   0.4377

### Scientific interpretation

The comparison demonstrates an important AI principle: a more
sophisticated probabilistic model does not necessarily minimize
reconstruction error. The VAE's advantage is its **regularized
probabilistic latent space**, rather than superior reconstruction alone.

------------------------------------------------------------------------

# Notebook 2 --- Shared Molecular Patterns

### Method

The notebook compares:

-   **Linear CCA**
-   **Neural shared projection**

CCA identifies linear projections that maximize correlation between two
molecular views. The neural model extends this idea using nonlinear
encoders to learn a shared representation.

### Result

Final first-component correlation:

``` text
Neural shared projection : 0.9672
Linear CCA               : 0.9519
```

The neural model therefore achieved a higher component-1 correlation on
the evaluated data.

The resulting shared spaces are visualized independently and coloured by
PAM50 only after fitting.

### Advanced AI concept

The key transition is from **linear statistical integration** to
**nonlinear neural representation alignment**.

------------------------------------------------------------------------

# Notebook 3 --- Latent Factor Discovery

### Method

**MOFA + attention integration**

MOFA models RNA, miRNA, and methylation as separate molecular views and
learns latent factors explaining variation within and across views.

An attention-based integration stage is then used to learn how strongly
each omics modality contributes to the integrated representation.

### Attention results

Mean learned attention:

``` text
RNA-seq     : 0.274
miRNA       : 0.521
Methylation : 0.205
```

Training time:

``` text
6.86 seconds
```

### Scientific interpretation

The attention mechanism provides an interpretable view of the
integration process. In this run, miRNA received the largest mean
attention contribution, followed by RNA-seq and methylation.

This illustrates how attention can move multi-omics integration beyond
simple concatenation or averaging by allowing the model to learn
**modality-specific weighting**.

------------------------------------------------------------------------

# Notebook 4 --- Patient Similarity

### Method

**Integrated similarity kernel + Graph Attention Network (GAT)**

The existing integrated patient similarity matrix is converted into a
patient graph.

Graph construction:

``` text
617 patients
3,085 original directed top-5-neighbour edges
6,077 reciprocal/self-loop message-passing edges
```

The GAT contains:

-   2 custom GAT layers
-   2 attention heads per layer
-   hidden dimension = 32
-   32-dimensional patient embedding

The model is trained using self-supervised link reconstruction. PAM50 is
excluded from graph construction and training.

### Results

  Representation     Silhouette     ARI     NMI
  ---------------- ------------ ------- -------
  Kernel PCA              0.133   0.271   0.403
  GAT PCA                 0.139   0.212   0.352

The GAT slightly improved silhouette geometry but did not outperform the
original kernel representation in ARI or NMI.

### Scientific interpretation

This is an intentional example of **model evaluation rather than model
promotion**. The GAT provides learned neighbourhood-aware and
interpretable patient embeddings, but greater model complexity does not
automatically produce stronger biological subtype agreement.

------------------------------------------------------------------------

# Notebook 5 --- Subtype Identification

### Method

**Multi-omics Transformer**

Each molecular modality is represented as a token:

``` text
[CLS] [RNA] [miRNA] [methylation]
```

Transformer configuration:

``` text
d_model = 64
layers  = 2
heads   = 4
```

The model uses self-attention to learn relationships among the molecular
views before producing a subtype prediction.

### Evaluation

The evaluated labelled cohort contained:

``` text
379 PAM50-labelled samples
242 training samples
76 test samples
```

Test performance:

``` text
Accuracy           : 0.803
Balanced accuracy  : 0.631
Macro precision    : 0.678
Macro recall       : 0.631
Macro F1           : 0.641
Weighted F1        : 0.780
```

### Modality ablation

  Model                         Accuracy   Balanced accuracy   Macro F1
  --------------------------- ---------- ------------------- ----------
  RNA only                         0.829               0.732      0.776
  RNA + miRNA                      0.776               0.575      0.588
  RNA + methylation                0.803               0.657      0.696
  RNA + miRNA + methylation        0.803               0.631      0.641

The RNA-only baseline outperformed the full Transformer on this
particular split.

### Scientific interpretation

The Transformer demonstrates **self-attention across omics modalities**,
but the results also show why advanced AI must be evaluated against
meaningful baselines and modality ablations.

Adding more molecular views does not guarantee better prediction. The
experiment therefore illustrates both the capability and the limitations
of Transformer-based multi-omics integration.

------------------------------------------------------------------------

# Advanced AI progression

The five notebooks deliberately build in complexity:

``` text
Nonlinear latent learning
        ↓
Neural representation alignment
        ↓
Attention-based multi-view integration
        ↓
Graph attention and message passing
        ↓
Transformer self-attention
```

This progression introduces several modern AI concepts without requiring
large GPU-based foundation models.

### 1. Representation learning

VAEs learn compressed nonlinear representations instead of relying only
on manually selected statistical projections.

### 2. Neural alignment

The neural shared projection demonstrates how nonlinear encoders can
align molecular representations across views.

### 3. Attention

Attention learns which molecular views contribute most strongly to an
integrated representation.

### 4. Graph neural networks

GATs treat patients as nodes and exploit learned relationships between
neighbouring patients.

### 5. Transformers

The final notebook treats omics modalities as tokens and uses
self-attention to model relationships between molecular views.

------------------------------------------------------------------------

# Reproducibility

The notebooks are designed to run from the repository root using
relative paths.

Expected project structure:

``` text
AI-Driven-MultiOmics-Workshop/
│
├── notebooks/
│   ├── 01_disease_subtyping_vae.ipynb
│   ├── 02_shared_molecular_patterns_cca.ipynb
│   ├── 03_latent_factors_mofa.ipynb
│   ├── 04_patient_similarity_kernel.ipynb
│   └── 05_subtype_identification_transformer.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│       └── figures/
│
└── README.md
```

Run notebooks from the project environment with the required Python
packages installed.

The notebooks save intermediate matrices, model outputs, metrics, and
figures under:

``` text
data/processed/
data/processed/figures/
```

------------------------------------------------------------------------

# Important methodological principles

### Unsupervised stages

The following representations are learned without PAM50 labels:

-   VAE latent representation
-   CCA/shared projection
-   MOFA factors
-   Attention-integrated representation
-   Patient similarity kernel
-   GAT patient embeddings

PAM50 is introduced afterward for biological interpretation and
evaluation.

### Baseline comparison

Advanced models are compared with simpler alternatives:

-   VAE vs AE
-   CCA vs neural shared projection
-   Kernel representation vs GAT
-   Full Transformer vs modality ablations

This avoids treating model complexity as evidence of improvement.

### Interpretability

The workshop emphasizes:

-   latent representations
-   variance/factor structure
-   modality attention weights
-   graph neighbourhood attention
-   Transformer attention
-   subtype-wise evaluation
-   modality ablation

The objective is not only prediction, but understanding **what
information the model uses and how molecular views interact**.

------------------------------------------------------------------------

# Workshop takeaway

The central message is that advanced AI for multi-omics is not simply
about applying the largest neural network available.

A useful multi-omics AI pipeline should:

1.  Learn meaningful representations.
2.  Integrate heterogeneous molecular views.
3.  Model biological relationships and patient similarity.
4.  Provide interpretable signals where possible.
5.  Be evaluated against appropriate baselines.
6.  Test whether additional modalities actually improve performance.
7.  Separate model fitting from downstream biological interpretation.
8.  Produce results that can generate biologically testable hypotheses.

The five notebooks provide a compact progression from classical
multi-omics integration to modern **attention-, graph-, and
Transformer-based AI approaches**, while remaining practical for a short
CPU-based hands-on workshop.

------------------------------------------------------------------------

## Outputs

The repository contains notebook-generated results including:

-   latent representations
-   factor scores
-   neural shared projections
-   patient embeddings
-   attention weights
-   model comparison tables
-   PAM50 evaluation metrics
-   reconstruction comparisons
-   publication-style figures

All generated figures and analysis outputs are organized under:

``` text
data/processed/figures/
```

------------------------------------------------------------------------

## License / usage

This repository is intended for educational and workshop use. Please
refer to the accompanying workshop documentation for methodological
context, dataset provenance, and interpretation of the analyses.
