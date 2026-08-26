# AI-Driven Multi-Omics Workshop

A workshop series on AI/ML-driven multi-omics analysis for computational biology.

## Notebook 1 — Disease Subtyping with a Variational Autoencoder

**Biological question:**  
Can a neural network learn a compact representation of breast cancer patients from multiple molecular layers and reveal biologically meaningful patient subgroups?

**Use case:** Unsupervised disease subtyping  
**Model:** Variational Autoencoder (VAE, PyTorch)  
**Dataset:** TCGA-BRCA (The Cancer Genome Atlas — Breast Invasive Carcinoma)  
**Hardware:** CPU only — no GPU required  

### Molecular data sources (UCSC Xena TCGA Hub)

| Modality | Xena dataset ID | Unit | Samples | Features |
|---|---|---|---|---|
| RNA-seq | `TCGA.BRCA.sampleMap/HiSeqV2` | log2(RSEM+1) | 1,218 | 20,530 |
| miRNA | `TCGA.BRCA.sampleMap/miRNA_HiSeq_gene` | log2(RPM+1) | 832 | 2,238 |
| Methylation | `TCGA.BRCA.sampleMap/HumanMethylation450` | beta [0,1] | 888 | 485,577 |
| Clinical | `TCGA.BRCA.sampleMap/BRCA_clinicalMatrix` | — | 1,247 | 193 |

### Final cohort (frozen)

- **617 unique primary tumour samples** (one per patient)
- Present in all three molecular modalities
- PAM50 labels available for 379 / 617 samples (used **post-hoc only**)

### Preprocessing summary

| Modality | Input | Output | Method |
|---|---|---|---|
| RNA-seq | 20,530 genes | 1,000 genes | Top 1,000 by variance (floor=0.01) |
| miRNA | 2,238 | 300 | Remove >50% sparse; top 300 by variance |
| Methylation | 485,577 CpGs | 1,000 CpGs | Chunked variance selection (cg* probes) |

### VAE architecture

```
Input: 2,300  →  Encoder: 256 → 64  →  Latent: μ, log σ² (10-D)
Latent: 10    →  Decoder: 64 → 256  →  Output: 2,300
```

- Optimizer: Adam (lr=1e-3), batch=64, early stopping (patience=20)
- Beta (KL weight): 1.0
- Best validation loss: 0.604

### Key results

| Metric | Value |
|---|---|
| Best val reconstruction loss | 0.593 |
| Best val KL loss | 0.011 |
| KL collapse | No |
| ARI (clusters vs PAM50) | 0.240 |
| NMI (clusters vs PAM50) | 0.386 |

## Repository structure

```
AI-Driven-MultiOmics-Workshop/
├── notebooks/
│   └── 01_disease_subtyping_vae.ipynb   ← main workshop notebook
├── data/
│   ├── raw/                              ← raw TCGA downloads (not tracked by Git)
│   ├── processed/
│   │   ├── TCGA_BRCA_cohort_metadata.tsv
│   │   ├── TCGA_BRCA_cohort_freeze.json
│   │   ├── TCGA_BRCA_methylation_selected_metadata.json
│   │   ├── TCGA_BRCA_preprocessing_provenance.json
│   │   ├── TCGA_BRCA_VAE_metadata.json
│   │   └── *.png                         ← result figures
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducing the analysis

1. Download raw TCGA-BRCA data from [UCSC Xena](https://xenabrowser.net) into `data/raw/`  
   (see `data/README.md` for exact dataset IDs and filenames)
2. Install dependencies: `pip install -r requirements.txt`
3. Open and run `notebooks/01_disease_subtyping_vae.ipynb` from top to bottom

> ⚠️ The methylation file is ~783 MB compressed / 2.7 GB uncompressed.  
> The notebook uses a memory-efficient chunked processing strategy — it does **not** load the full file into RAM.

## Important notes

- PAM50 subtype labels are used **only** for post-hoc biological interpretation.  
  They are **not** used during preprocessing, VAE training, or clustering.
- Raw TCGA data files are **not** tracked by Git (see `.gitignore`).
- Processed large matrices (`.tsv`) are also excluded from Git.  
  Small provenance JSON files and result figures are tracked.
