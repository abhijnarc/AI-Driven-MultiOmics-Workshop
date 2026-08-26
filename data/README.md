# data/

This directory holds TCGA-BRCA data for the workshop.  
**Raw data files are not tracked by Git** — they must be downloaded locally.

---

## data/raw/   (not in Git)

Download the following files from the [UCSC Xena TCGA Hub](https://tcga.xenahubs.net)  
and save them to `data/raw/` with the filenames shown below.

| Local filename | Xena dataset ID | Download URL |
|---|---|---|
| `TCGA_BRCA_RNAseq.tsv.gz` | `TCGA.BRCA.sampleMap/HiSeqV2` | `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2.gz` |
| `TCGA_BRCA_miRNA.tsv.gz` | `TCGA.BRCA.sampleMap/miRNA_HiSeq_gene` | `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FmiRNA_HiSeq_gene.gz` |
| `TCGA_BRCA_methylation.tsv.gz` | `TCGA.BRCA.sampleMap/HumanMethylation450` | `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHumanMethylation450.gz` |
| `TCGA_BRCA_clinical.tsv` | `TCGA.BRCA.sampleMap/BRCA_clinicalMatrix` | `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix` |

> The notebook (`notebooks/01_disease_subtyping_vae.ipynb`) handles downloading automatically  
> if the files are not present. It will also decompress the `.gz` files where needed.

---

## data/processed/   (partially tracked by Git)

The following **small provenance/metadata files** are tracked:

| File | Description |
|---|---|
| `TCGA_BRCA_cohort_freeze.json` | Frozen cohort definition: 617 samples, inclusion rule, excluded samples |
| `TCGA_BRCA_cohort_metadata.tsv` | Per-sample: sample_id, patient_id, sample_type, PAM50_subtype |
| `TCGA_BRCA_methylation_selected_metadata.json` | Methylation feature selection provenance |
| `TCGA_BRCA_preprocessing_provenance.json` | Full preprocessing audit for all 3 modalities |
| `TCGA_BRCA_VAE_metadata.json` | VAE architecture, training config, final losses |
| `TCGA_BRCA_VAE_UMAP_PAM50.png` | UMAP coloured by PAM50 (post-hoc) |
| `TCGA_BRCA_VAE_UMAP_clusters.png` | UMAP coloured by K-means clusters |
| `TCGA_BRCA_VAE_contingency.png` | Contingency heatmap: clusters vs PAM50 |
| `TCGA_BRCA_VAE_cluster_composition.png` | Stacked bar: PAM50 composition per cluster |

The following **large files are NOT tracked** (regenerate by running the notebook):

| File | Description | Why excluded |
|---|---|---|
| `TCGA_BRCA_RNA_selected.tsv` | 617 × 1,000 RNA features | ~5.6 MB, regenerable |
| `TCGA_BRCA_miRNA_selected.tsv` | 617 × 300 miRNA features | ~1.7 MB, regenerable |
| `TCGA_BRCA_methylation_selected.tsv` | 617 × 1,000 CpG features | ~5.6 MB, regenerable |
| `TCGA_BRCA_VAE_latent.tsv` | 617 × 10 latent representation | regenerable |
| `TCGA_BRCA_VAE_best.pt` | PyTorch model checkpoint | ~4.7 MB binary |
