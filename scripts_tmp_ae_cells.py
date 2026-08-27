"""Temporary runner: executes the exact code of each new notebook cell,
captures stdout per cell, and saves results so they can be embedded
into notebooks/01_disease_subtyping_vae.ipynb."""
import io, json, sys, contextlib
import matplotlib
matplotlib.use("Agg")

CELL_RESULTS = {}

def run_cell(name, fn):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
        CELL_RESULTS[name] = {"outputs": buf.getvalue(), "error": None}
        print(f"[OK] {name}")
    except Exception as e:
        import traceback
        CELL_RESULTS[name] = {"outputs": buf.getvalue(),
                              "error": traceback.format_exc()}
        print(f"[FAIL] {name}: {e}")

# ════════════════════════════════════════════════════════════════════
# Cell 1 (code): setup + data loading + shared split/scaling
# ════════════════════════════════════════════════════════════════════
def cell_setup():
    global PROCESSED_DIR, FIG_DIR, rna_df, mirna_df, meth_df, cohort_meta
    global frozen_ids, idx_train, idx_val, X_all, X_tensor, INPUT_DIM
    global LATENT_DIM, VAE_SEED, BATCH_SIZE, train_loader, val_loader
    global X_train_t, X_val_t, vae_metadata
    import random, time, json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    NOTEBOOK_DIR = Path(".").resolve()
    PROJECT_DIR = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
    FIG_DIR = PROCESSED_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    VAE_SEED = 42
    random.seed(VAE_SEED); np.random.seed(VAE_SEED); torch.manual_seed(VAE_SEED)

    rna_df   = pd.read_csv(PROCESSED_DIR / "TCGA_BRCA_RNA_selected.tsv",   sep="\t", index_col=0)
    mirna_df = pd.read_csv(PROCESSED_DIR / "TCGA_BRCA_miRNA_selected.tsv", sep="\t", index_col=0)
    meth_df  = pd.read_csv(PROCESSED_DIR / "TCGA_BRCA_methylation_selected.tsv", sep="\t", index_col=0)
    cohort_meta = pd.read_csv(PROCESSED_DIR / "TCGA_BRCA_cohort_metadata.tsv", sep="\t", index_col="sample_id")
    frozen_ids = cohort_meta.index.tolist()

    assert list(rna_df.index) == list(mirna_df.index) == list(meth_df.index) == frozen_ids

    with open(PROCESSED_DIR / "TCGA_BRCA_VAE_metadata.json") as fh:
        vae_metadata = json.load(fh)

    # Same split & scaling as the VAE section (identical seed → identical split)
    VAL_FRAC, BATCH_SIZE = 0.20, 64
    sample_array = np.array(frozen_ids)
    idx_all = np.arange(len(sample_array))
    idx_train, idx_val = train_test_split(idx_all, test_size=VAL_FRAC,
                                          random_state=VAE_SEED, shuffle=True)
    train_ids = sample_array[idx_train].tolist()

    def fit_scale(df, tr_ids):
        sc = StandardScaler().fit(df.loc[tr_ids].values)
        return sc.transform(df.values).astype(np.float32)

    X_all = np.concatenate([fit_scale(rna_df, train_ids),
                            fit_scale(mirna_df, train_ids),
                            fit_scale(meth_df, train_ids)], axis=1)
    INPUT_DIM = X_all.shape[1]
    LATENT_DIM = 10

    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    X_train_t, X_val_t = X_tensor[idx_train], X_tensor[idx_val]
    train_loader = DataLoader(TensorDataset(X_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t),   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Figures dir : {FIG_DIR}")
    print(f"Cohort      : {len(frozen_ids)} samples | input dim {INPUT_DIM} | latent {LATENT_DIM}")
    print(f"Train/val   : {len(idx_train)}/{len(idx_val)} (same split & scaling as VAE)")

# ════════════════════════════════════════════════════════════════════
# Cell 2 (code): define + train lightweight AE
# ════════════════════════════════════════════════════════════════════
def cell_ae_train():
    import time, torch, torch.nn as nn, torch.nn.functional as F

    class AE(nn.Module):
        """Same encoder/decoder topology as the VAE, but deterministic bottleneck (no mu/logvar, no KL)."""
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
                nn.Linear(64, 256),         nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, input_dim),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    torch.manual_seed(VAE_SEED)
    ae = AE(INPUT_DIM, LATENT_DIM)
    n_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    print("AE architecture:")
    print(ae)
    print(f"\nTrainable parameters: {n_params:,}")

    LR, MAX_EPOCHS, PATIENCE = 1e-3, 100, 15
    AE_CKPT = PROCESSED_DIR / "TCGA_BRCA_AE_best.pt"
    opt = torch.optim.Adam(ae.parameters(), lr=LR)
    ae_history = {"train_recon": [], "val_recon": []}
    best_val, best_epoch, patience_ct = float("inf"), 0, 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        ae.train()
        tr = 0.0
        for (xb,) in train_loader:
            opt.zero_grad()
            loss = F.mse_loss(ae(xb), xb, reduction="mean")
            loss.backward(); opt.step()
            tr += loss.item()
        ae_history["train_recon"].append(tr / len(train_loader))

        ae.eval()
        va = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                va += F.mse_loss(ae(xb), xb, reduction="mean").item()
        va /= len(val_loader)
        ae_history["val_recon"].append(va)

        if va < best_val:
            best_val, best_epoch, patience_ct = va, epoch, 0
            torch.save({"epoch": epoch, "model_state": ae.state_dict(), "val_recon": va}, AE_CKPT)
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train recon={ae_history['train_recon'][-1]:.5f} "
                  f"| val recon={va:.5f}")

    elapsed = time.time() - t0
    ckpt = torch.load(AE_CKPT, weights_only=True)
    ae.load_state_dict(ckpt["model_state"]); ae.eval()
    globals()["ae"] = ae; globals()["ae_history"] = ae_history
    globals()["ae_best_epoch"] = best_epoch; globals()["ae_elapsed"] = elapsed
    globals()["ae_n_params"] = n_params
    print(f"\nTraining finished: {len(ae_history['train_recon'])} epochs in {elapsed:.1f}s")
    print(f"Best epoch: {best_epoch} | best val recon (MSE): {best_val:.5f}")

# ════════════════════════════════════════════════════════════════════
# Cell 3 (code): VAE vs AE reconstruction-loss comparison
# ════════════════════════════════════════════════════════════════════
def cell_compare():
    import json, torch, torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np

    # VAE reconstruction loss on the SAME validation set (deterministic, mu path)
    ckpt_vae = torch.load(PROCESSED_DIR / "TCGA_BRCA_VAE_best.pt", weights_only=True)
    from pathlib import Path
    # Rebuild VAE exactly as in Part 15
    class Encoder(torch.nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(),
                torch.nn.Linear(256, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU())
            self.fc_mu = torch.nn.Linear(64, latent_dim)
            self.fc_logvar = torch.nn.Linear(64, latent_dim)
        def forward(self, x):
            h = self.net(x); return self.fc_mu(h), self.fc_logvar(h)
    class Decoder(torch.nn.Module):
        def __init__(self, latent_dim, output_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU(),
                torch.nn.Linear(64, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(),
                torch.nn.Linear(256, output_dim))
        def forward(self, z): return self.net(z)
    class VAE(torch.nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = Encoder(input_dim, latent_dim)
            self.decoder = Decoder(latent_dim, input_dim)
        def forward(self, x):
            mu, logvar = self.encoder(x)
            return self.decoder(mu), mu, logvar

    vae = VAE(INPUT_DIM, LATENT_DIM)
    vae.load_state_dict(ckpt_vae["model_state"]); vae.eval()
    globals()["vae"] = vae

    with torch.no_grad():
        vr = 0.0
        for (xb,) in val_loader:
            x_hat, _, _ = vae(xb)
            vr += F.mse_loss(x_hat, xb, reduction="mean").item()
        vae_val_recon = vr / len(val_loader)

        ar = 0.0
        for (xb,) in val_loader:
            ar += F.mse_loss(ae(xb), xb, reduction="mean").item()
        ae_val_recon = ar / len(val_loader)

    comparison = {
        "vae_val_recon_mse": round(vae_val_recon, 6),
        "ae_val_recon_mse":  round(ae_val_recon, 6),
        "delta_ae_minus_vae": round(ae_val_recon - vae_val_recon, 6),
        "pct_difference": round(100 * (ae_val_recon - vae_val_recon) / vae_val_recon, 2),
        "vae_best_epoch": ckpt_vae["epoch"],
        "ae_best_epoch": ae_best_epoch,
        "vae_params": None, "ae_params": ae_n_params,
    }
    globals()["comparison"] = comparison

    print("Reconstruction loss on validation set (MSE, mean over elements)")
    print("=" * 55)
    print(f"  VAE : {vae_val_recon:.5f}   (best epoch {ckpt_vae['epoch']})")
    print(f"  AE  : {ae_val_recon:.5f}   (best epoch {ae_best_epoch})")
    d = ae_val_recon - vae_val_recon
    print(f"  Δ (AE − VAE): {d:+.5f} ({comparison['pct_difference']:+.2f}%)")
    print()
    if abs(d) < 0.02 * vae_val_recon:
        print("Interpretation: reconstruction quality is essentially comparable.")
        print("The VAE's advantage lies in its smooth, probabilistic latent space")
        print("(regularised by the KL term toward N(0,I)), not in lower MSE.")
    elif d > 0:
        print("Interpretation: the AE reconstructs slightly better — expected, since")
        print("it spends ALL of its capacity on reconstruction with no KL constraint.")
    else:
        print("Interpretation: the VAE reconstructs slightly better despite the KL")
        print("constraint acting as a regulariser against overfitting.")

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["VAE\n(KL-regularised)", "AE\n(deterministic)"],
                  [vae_val_recon, ae_val_recon],
                  color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.5)
    for b, v in zip(bars, [vae_val_recon, ae_val_recon]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.0002, f"{v:.5f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Validation reconstruction MSE")
    ax.set_title("VAE vs plain Autoencoder\nsame inputs, same latent dim (10), same split")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "VAE_vs_AE_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: data/processed/figures/VAE_vs_AE_reconstruction.png")

    with open(PROCESSED_DIR / "TCGA_BRCA_VAE_vs_AE_comparison.json", "w") as fh:
        json.dump(comparison, fh, indent=2)
    print("✅ Saved: data/processed/TCGA_BRCA_VAE_vs_AE_comparison.json")

# ════════════════════════════════════════════════════════════════════
# Cell 4 (code): UMAP of both latent spaces + PAM50 post-hoc overlay
# ════════════════════════════════════════════════════════════════════
def cell_umap():
    import torch, numpy as np, pandas as pd
    import matplotlib.pyplot as plt
    import umap as umap_lib
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.cluster import KMeans

    # Latents (no labels used anywhere in embedding)
    with torch.no_grad():
        z_vae = vae.encoder(X_tensor)[0].numpy()          # mu, (617, 10)
        z_ae  = ae.encoder(X_tensor).numpy()              # (617, 10)

    pam50_labels = cohort_meta["PAM50_subtype"].reindex(frozen_ids).fillna("Unknown")

    reducer_vae = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    reducer_ae  = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    emb_vae = reducer_vae.fit_transform(z_vae)
    emb_ae  = reducer_ae.fit_transform(z_ae)

    SUBTYPE_ORDER  = ["LumA", "LumB", "Basal", "Her2", "Normal", "Unknown"]
    SUBTYPE_COLORS = {"LumA": "#4C72B0", "LumB": "#DD8452", "Basal": "#C44E52",
                      "Her2": "#8172B3", "Normal": "#55A868", "Unknown": "#CCCCCC"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, (emb, name) in enumerate([(emb_vae, "VAE"), (emb_ae, "AE")]):
        axes[row, 0].scatter(emb[:, 0], emb[:, 1], s=10, color="steelblue",
                             alpha=0.5, rasterized=True)
        axes[row, 0].set_title(f"{name} latent space (UMAP)\nunsupervised structure, no labels",
                               fontsize=11)
        axes[row, 0].set_xlabel("UMAP 1"); axes[row, 0].set_ylabel("UMAP 2")
        axes[row, 0].spines[["top", "right"]].set_visible(False)

        for st in SUBTYPE_ORDER:
            mask = pam50_labels.values == st
            if mask.sum() == 0:
                continue
            alpha = 0.3 if st == "Unknown" else 0.75
            size  = 8   if st == "Unknown" else 14
            axes[row, 1].scatter(emb[mask, 0], emb[mask, 1], s=size,
                                 color=SUBTYPE_COLORS[st], alpha=alpha,
                                 label=f"{st} (n={mask.sum()})", rasterized=True,
                                 zorder=2 if st != "Unknown" else 1)
        axes[row, 1].set_title(f"{name} latent space (UMAP)\ncoloured by PAM50 (post-hoc reference only)",
                               fontsize=11)
        axes[row, 1].set_xlabel("UMAP 1"); axes[row, 1].set_ylabel("UMAP 2")
        axes[row, 1].legend(title="PAM50 subtype", bbox_to_anchor=(1.01, 1),
                            loc="upper left", fontsize=8, title_fontsize=9)
        axes[row, 1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Nonlinear Representation Learning Comparison — VAE vs Autoencoder\n"
                 "617 TCGA-BRCA patients · 2,300 multi-omics features · 10-D latent",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "VAE_vs_AE_UMAP_PAM50.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: data/processed/figures/VAE_vs_AE_UMAP_PAM50.png")

    # Post-hoc quantitative check: k-means (k=5) vs PAM50, labelled samples only
    rows = []
    has_lbl = pam50_labels.values != "Unknown"
    for z, name in [(z_vae, "VAE"), (z_ae, "AE")]:
        km = KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(z)
        ari = adjusted_rand_score(pam50_labels.values[has_lbl], km[has_lbl])
        nmi = normalized_mutual_info_score(pam50_labels.values[has_lbl], km[has_lbl])
        rows.append({"model": name, "ARI_vs_PAM50": round(ari, 4),
                     "NMI_vs_PAM50": round(nmi, 4)})
    posthoc = pd.DataFrame(rows).set_index("model")
    globals()["posthoc"] = posthoc
    print("\nPost-hoc PAM50 agreement (k-means k=5 on 10-D latent; labels NOT used in fitting):")
    print(posthoc.to_string())
    print("\n⚠️  PAM50 used strictly post-hoc as an external biological reference.")

    posthoc.to_csv(PROCESSED_DIR / "TCGA_BRCA_VAE_vs_AE_PAM50_posthoc.tsv", sep="\t")
    print("✅ Saved: data/processed/TCGA_BRCA_VAE_vs_AE_PAM50_posthoc.tsv")

# ════════════════════════════════════════════════════════════════════
run_cell("setup", cell_setup)
run_cell("ae_train", cell_ae_train)
run_cell("compare", cell_compare)
if "--with-umap" in sys.argv:
    run_cell("umap", cell_umap)

with open("_cell_results.json", "w", encoding="utf-8") as fh:
    json.dump(CELL_RESULTS, fh, indent=1)
print("\nDone — results written to _cell_results.json")