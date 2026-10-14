"""Analysis of the v3 results (corrected protocol, original + de-duplicated splits).

Reads only the per-run records the Kaggle notebook wrote
(results/v3/raw/runs/<split>/<backbone>_seed<seed>.json plus data/, latency and
TFLite files). Nothing is retrained.

Estimand. For each backbone, the quantity of interest is the expected test
accuracy of the *training procedure* (fixed data split, random seed drawn at
random) on images drawn from the same distribution as the test set. Two
sources of uncertainty are therefore propagated: training randomness (5 seeds)
and test-set sampling (105/106 images). The hierarchical bootstrap resamples
both; image-level paired tests treat test images as the independent units
(per-image correctness averaged over seeds); the GEE logistic model clusters
the 5 seed-level predictions of each image and reports odds ratios.

Run from the repo root:  python scripts/analysis_v3.py
Outputs: results/v3/analysis/{tables,figures}/ and key_numbers.json
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, matthews_corrcoef)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "v3" / "raw"
OUT = ROOT / "results" / "v3" / "analysis"
FIG, TAB = OUT / "figures", OUT / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

MODELS = ["MobileNetV3Small", "EfficientNetB0", "DenseNet121"]
SPLITS = ["original", "dedup"]
SPLIT_LABEL = {"original": "Original split", "dedup": "De-duplicated split"}
SEEDS = [42, 1042, 2042, 3042, 4042]
# Categorical slots 1-3 of the validated reference palette (all-pairs CVD-safe).
COLORS = {"MobileNetV3Small": "#2a78d6", "EfficientNetB0": "#eb6834", "DenseNet121": "#1baf7a"}
CLASS_LABELS = ["Diseased leaf", "Diseased plant", "Fresh leaf", "Fresh plant"]
K = 4
N_BOOT = 10000
RNG = np.random.default_rng(2026)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def macro_f1_fast(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean()


def holm(p):
    return multipletests(p, method="holm")[1]


# ----------------------------------------------------------------- load
R = {}                               # (split, model, seed) -> run record
for sp in SPLITS:
    for m in MODELS:
        for s in SEEDS:
            R[sp, m, s] = json.loads((RAW / "runs" / sp / f"{m}_seed{s}.json").read_text())
PATHS = {sp: R[sp, MODELS[0], SEEDS[0]]["test_relpaths"] for sp in SPLITS}
Y = {sp: np.array(R[sp, MODELS[0], SEEDS[0]]["y_true"]) for sp in SPLITS}
for (sp, m, s), r in R.items():      # every run of a split saw the same images in the same order
    assert r["test_relpaths"] == PATHS[sp] and r["y_true"] == Y[sp].tolist(), (sp, m, s)
    assert r.get("bn_frozen_in_finetune") is True, (sp, m, s)
PRED = {(sp, m): np.array([R[sp, m, s]["y_pred"] for s in SEEDS]) for sp in SPLITS for m in MODELS}
PROB = {(sp, m): np.array([R[sp, m, s]["y_prob"] for s in SEEDS]) for sp in SPLITS for m in MODELS}
CORR = {k: (v == Y[k[0]][None, :]).astype(float) for k, v in PRED.items()}
key = {"n_test": {sp: int(len(Y[sp])) for sp in SPLITS}}

# ------------------------------------------------------ 1. dataset and duplicate audit
manifest = pd.read_csv(RAW / "data" / "manifest.csv")
pairs_dup = pd.read_csv(RAW / "data" / "duplicate_pairs.csv")
counts = pd.read_csv(RAW / "data" / "class_counts.csv")
counts.to_csv(TAB / "class_counts.csv", index=False)
dup_by_split = (pairs_dup.assign(pair=lambda d: d[["split_a", "split_b"]].apply(lambda r: "-".join(sorted(r)), axis=1))
                .groupby("pair").agg(n_pairs=("i", "size"), identical_bytes=("identical_bytes", "sum"),
                                     cross_class=("cross_class", "sum")).reset_index())
dup_by_split.to_csv(TAB / "duplicate_pairs_by_split.csv", index=False)
removed = (manifest[~manifest.keep_dedup].groupby(["split", "cls"]).size()
           .rename("n_removed").reset_index())
removed.to_csv(TAB / "dedup_removed_images.csv", index=False)
n_groups = manifest.groupby("group").size()
key["duplicates"] = dict(
    n_images=int(len(manifest)), n_pairs=int(len(pairs_dup)),
    n_cross_split_pairs=int(pairs_dup.cross_split.sum()),
    n_cross_class_pairs=int(pairs_dup.cross_class.sum()),
    n_identical_byte_pairs=int(pairs_dup.identical_bytes.sum()),
    n_groups_size_gt1=int((n_groups > 1).sum()),
    pairs_by_split=dup_by_split.set_index("pair").n_pairs.to_dict(),
    removed_by_split=manifest[~manifest.keep_dedup].split.value_counts().to_dict(),
    counts={v: counts[counts.version == v].set_index("split").drop(columns="version").sum(1).to_dict()
            for v in SPLITS})

# ------------------------------------------------ 2. per-run metrics
rows = []
for (sp, m, s), r in R.items():
    p, y = np.array(r["y_pred"]), Y[sp]
    h1, h2 = r["history_phase1"], r["history_phase2"]
    rows.append(dict(
        split=sp, model=m, seed=s, accuracy=accuracy_score(y, p),
        macro_f1=f1_score(y, p, average="macro"), balanced_accuracy=balanced_accuracy_score(y, p),
        mcc=matthews_corrcoef(y, p), cohen_kappa=cohen_kappa_score(y, p),
        selected_phase=r["selected_phase"], head_epochs=len(h1["loss"]), ft_epochs=len(h2["loss"]),
        head_best_epoch=int(np.argmin(h1["val_loss"])) + 1, ft_best_epoch=int(np.argmin(h2["val_loss"])) + 1,
        val_loss_phase1=r["val_loss_phase1"], val_loss_phase2=r["val_loss_phase2"],
        val_acc_selected=r["val_acc_selected"], train_loss_end_phase1=h1["loss"][-1],
        train_loss_first_ft_epoch=h2["loss"][0], train_time_min=r["train_time_s"] / 60,
        params=r["params"], weights_mb=r["weights_mb"], n_train=r["n_train"], n_val=r["n_val"],
        n_test=r["n_test"]))
per_run = pd.DataFrame(rows)
per_run.to_csv(TAB / "per_run_metrics.csv", index=False)
key["phase2_selected"] = int((per_run.selected_phase == 2).sum())
key["head_hit_cap"] = int((per_run.head_epochs == 20).sum())


# ------------------------------------------------ 3. summary with three kinds of interval
def hier_boot(sp, m, metric, n=N_BOOT):
    """Two-level bootstrap: resample seeds, then test images (same images for every resampled seed)."""
    out = np.empty(n)
    C, P, y = CORR[sp, m], PRED[sp, m], Y[sp]
    for b in range(n):
        si = RNG.integers(0, len(SEEDS), len(SEEDS))
        ii = RNG.integers(0, len(y), len(y))
        out[b] = C[si][:, ii].mean() if metric == "accuracy" else \
            np.mean([macro_f1_fast(y[ii], P[s][ii]) for s in si])
    return np.percentile(out, [2.5, 97.5])


summ = []
for sp in SPLITS:
    for m in MODELS:
        g = per_run[(per_run.split == sp) & (per_run.model == m)]
        row = dict(split=sp, model=m)
        for met in ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "cohen_kappa"]:
            v = g[met].values
            row[f"{met}_mean"], row[f"{met}_sd"] = v.mean(), v.std(ddof=1)
            row[f"{met}_min"], row[f"{met}_max"] = v.min(), v.max()
        for met in ["accuracy", "macro_f1"]:
            v = g[met].values
            tq = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            row[f"{met}_seedCI_lo"], row[f"{met}_seedCI_hi"] = v.mean() - tq, v.mean() + tq
            row[f"{met}_hierCI_lo"], row[f"{met}_hierCI_hi"] = hier_boot(sp, m, met)
        row["phase2_selected"] = int((g.selected_phase == 2).sum())
        row["train_time_min_mean"] = g.train_time_min.mean()
        row["params_million"] = g.params.iloc[0] / 1e6
        row["weights_mb"] = g.weights_mb.iloc[0]
        summ.append(row)
summary = pd.DataFrame(summ)
summary.to_csv(TAB / "metrics_summary.csv", index=False)
key["summary"] = {f"{r.split}/{r.model}": {k: (round(v, 6) if isinstance(v, float) else v)
                                           for k, v in r._asdict().items() if k not in ("split", "model")}
                  for r in summary.itertuples(index=False)}

# ------------------------------------------------ 4. pairwise comparisons within each split
pair_rows = []
for sp in SPLITS:
    p_img = {m: CORR[sp, m].mean(0) for m in MODELS}
    fr = stats.friedmanchisquare(*[p_img[m] for m in MODELS])
    key.setdefault("friedman", {})[sp] = dict(statistic=float(fr.statistic), p=float(fr.pvalue))
    n = len(Y[sp])
    for a, b in combinations(MODELS, 2):
        d = p_img[a] - p_img[b]
        obs = d.mean()
        perm = (RNG.choice([-1.0, 1.0], size=(100000, n)) * d[None, :]).mean(1)
        p_perm = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (len(perm) + 1)
        nz = d[d != 0]
        w_p = stats.wilcoxon(nz).pvalue if len(nz) else 1.0
        diffs = np.empty(N_BOOT)
        for k in range(N_BOOT):
            si = RNG.integers(0, len(SEEDS), len(SEEDS))
            ii = RNG.integers(0, n, n)
            diffs[k] = CORR[sp, a][si][:, ii].mean() - CORR[sp, b][si][:, ii].mean()
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        mc_p, disc = [], []
        for si in range(len(SEEDS)):
            ca, cb = CORR[sp, a][si].astype(bool), CORR[sp, b][si].astype(bool)
            n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
            disc.append((n01, n10))
            mc_p.append(1.0 if n01 + n10 == 0 else mcnemar([[0, n01], [n10, 0]], exact=True).pvalue)
        seed_diff = np.array([per_run.query("split==@sp and model==@a and seed==@s").accuracy.iloc[0]
                              - per_run.query("split==@sp and model==@b and seed==@s").accuracy.iloc[0]
                              for s in SEEDS])
        pair_rows.append(dict(
            split=sp, model_a=a, model_b=b, acc_diff_mean=obs, acc_diff_hierCI_lo=lo, acc_diff_hierCI_hi=hi,
            images_a_better=int(np.sum(d > 0)), images_b_better=int(np.sum(d < 0)),
            images_tied=int(np.sum(d == 0)), image_perm_p=p_perm, image_wilcoxon_p=w_p,
            seed_wins_a=int(np.sum(seed_diff > 0)), seed_wins_b=int(np.sum(seed_diff < 0)),
            pooled_discordant_a_only=sum(x for x, _ in disc), pooled_discordant_b_only=sum(y_ for _, y_ in disc),
            mcnemar_fisher_p=stats.combine_pvalues(mc_p, method="fisher").pvalue))
pairs = pd.DataFrame(pair_rows)
for col in ["image_perm_p", "image_wilcoxon_p", "mcnemar_fisher_p"]:
    pairs[f"{col}_holm"] = pairs.groupby("split")[col].transform(holm)
pairs.to_csv(TAB / "pairwise_tests.csv", index=False)
key["pairwise"] = pairs.round(6).to_dict(orient="records")


# ------------------------------------------------ 5. GEE logistic models (effect sizes)
def long_frame(sp, images=None):
    rows = []
    for m in MODELS:
        for si, s in enumerate(SEEDS):
            for i, rel in enumerate(PATHS[sp]):
                if images is None or rel in images:
                    rows.append(dict(split=sp, model=m, seed=s, image=rel, correct=int(CORR[sp, m][si, i])))
    return pd.DataFrame(rows)


gee_rows = []
for sp in SPLITS:
    df = long_frame(sp)
    df["model"] = pd.Categorical(df.model, MODELS)
    for ref in ["MobileNetV3Small", "EfficientNetB0"]:
        f = f"correct ~ C(model, Treatment('{ref}'))"
        res = smf.gee(f, "image", df, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = res.conf_int()
        for term in res.params.index[1:]:
            other = term.split("[T.")[1].rstrip("]")
            if ref == "EfficientNetB0" and other != "DenseNet121":
                continue
            gee_rows.append(dict(split=sp, reference=ref, model=other,
                                 odds_ratio=np.exp(res.params[term]),
                                 or_ci_lo=np.exp(ci.loc[term, 0]), or_ci_hi=np.exp(ci.loc[term, 1]),
                                 p=res.pvalues[term], working_corr=float(res.cov_struct.dep_params)))
gee = pd.DataFrame(gee_rows)
gee["p_holm"] = gee.groupby("split").p.transform(holm)
gee.to_csv(TAB / "gee_model_comparison.csv", index=False)
key["gee_models"] = gee.round(6).to_dict(orient="records")

# ------------------------------------------------ 6. original vs de-duplicated split (shared test images)
shared = sorted(set(PATHS["original"]) & set(PATHS["dedup"]))
idx = {sp: np.array([PATHS[sp].index(p) for p in shared]) for sp in SPLITS}
dd_rows = []
for m in MODELS:
    co, cd = CORR["original", m][:, idx["original"]], CORR["dedup", m][:, idx["dedup"]]
    d = co.mean(0) - cd.mean(0)
    diffs = np.empty(N_BOOT)
    for k in range(N_BOOT):
        si, sj = RNG.integers(0, 5, 5), RNG.integers(0, 5, 5)   # seeds are independent across splits
        ii = RNG.integers(0, len(shared), len(shared))
        diffs[k] = co[si][:, ii].mean() - cd[sj][:, ii].mean()
    perm = (RNG.choice([-1.0, 1.0], size=(100000, len(shared))) * d[None, :]).mean(1)
    dd_rows.append(dict(model=m, n_shared_images=len(shared), acc_original=co.mean(), acc_dedup=cd.mean(),
                        diff=co.mean() - cd.mean(), diff_CI_lo=np.percentile(diffs, 2.5),
                        diff_CI_hi=np.percentile(diffs, 97.5),
                        image_perm_p=(np.sum(np.abs(perm) >= abs(d.mean()) - 1e-12) + 1) / (len(perm) + 1)))
split_cmp = pd.DataFrame(dd_rows)
split_cmp.to_csv(TAB / "original_vs_dedup.csv", index=False)
key["original_vs_dedup"] = split_cmp.round(6).to_dict(orient="records")

df = pd.concat([long_frame(sp, set(shared)) for sp in SPLITS])
df["model"] = pd.Categorical(df.model, MODELS)
df["split"] = pd.Categorical(df.split, SPLITS)
res = smf.gee("correct ~ C(model) * C(split)", "image", df, family=sm.families.Binomial(),
              cov_struct=sm.cov_struct.Exchangeable()).fit()
wald = res.wald_test_terms(scalar=True)
gee_split = wald.table.reset_index().rename(columns={"index": "term"})
gee_split.to_csv(TAB / "gee_split_interaction.csv", index=False)
key["gee_split_terms"] = {r.term: dict(stat=float(r.statistic), p=float(r.pvalue), df=int(r.df_constraint))
                          for r in gee_split.itertuples()}

# duplicated test images in the ORIGINAL split: are they easier?
dup_test = set(manifest[(manifest.split == "test") & manifest.group.isin(
    manifest.groupby("group").filter(lambda g: (len(g) > 1) and (g.split != "test").any()).group)].relpath)
mask = np.array([p in dup_test for p in PATHS["original"]])
key["original_test_images_with_cross_split_duplicate"] = int(mask.sum())
key["acc_on_duplicated_vs_clean_test_images"] = {
    m: dict(duplicated=float(CORR["original", m][:, mask].mean()), clean=float(CORR["original", m][:, ~mask].mean()))
    for m in MODELS}

# ------------------------------------------------ 7. per-class F1 and confusion
pc_rows = []
for sp in SPLITS:
    for m in MODELS:
        arr = np.array([f1_score(Y[sp], PRED[sp, m][i], average=None, labels=range(K)) for i in range(5)])
        for c in range(K):
            pc_rows.append(dict(split=sp, model=m, cls=CLASS_LABELS[c], f1_mean=arr[:, c].mean(),
                                f1_sd=arr[:, c].std(ddof=1)))
per_class = pd.DataFrame(pc_rows)
per_class.to_csv(TAB / "per_class_f1.csv", index=False)

for sp in SPLITS:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    for ax, m in zip(axes, MODELS):
        cm = sum(np.bincount(Y[sp] * K + PRED[sp, m][i], minlength=K * K).reshape(K, K) for i in range(5))
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{norm[i, j]*100:.1f}%\n({cm[i, j]})", ha="center", va="center", fontsize=6.3,
                        color="white" if norm[i, j] > 0.6 else INK)
        ax.set_title(m, color=INK)
        ax.set_xticks(range(K), [c.replace(" ", "\n") for c in CLASS_LABELS], fontsize=6.8)
        ax.set_yticks(range(K), CLASS_LABELS if m == MODELS[0] else [""] * K, fontsize=6.8)
        ax.set_xlabel("Predicted")
        if m == MODELS[0]:
            ax.set_ylabel("True")
    save(fig, f"fig_confusion_{sp}")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), constrained_layout=True)
for ax, sp in zip(axes, SPLITS):
    sub = per_class[per_class.split == sp]
    mat = sub.pivot(index="model", columns="cls", values="f1_mean").loc[MODELS, CLASS_LABELS]
    sdm = sub.pivot(index="model", columns="cls", values="f1_sd").loc[MODELS, CLASS_LABELS]
    im = ax.imshow(mat.values, cmap="Blues", vmin=0.85, vmax=1.0, aspect="auto")
    ax.grid(False)
    for i in range(3):
        for j in range(K):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.3f}\n±{sdm.values[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.95 else INK)
    ax.set_xticks(range(K), [c.replace(" ", "\n") for c in CLASS_LABELS], fontsize=7)
    ax.set_yticks(range(3), MODELS if sp == "original" else [""] * 3)
    ax.set_title(SPLIT_LABEL[sp])
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02).set_label("Per-class F1 (mean ± SD, 5 seeds)")
save(fig, "fig_per_class_f1")

# ------------------------------------------------ 8. accuracy by seed + model differences
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, constrained_layout=True)
for ax, sp in zip(axes, SPLITS):
    for j, m in enumerate(MODELS):
        v = per_run[(per_run.split == sp) & (per_run.model == m)].accuracy.values * 100
        ax.scatter(np.full(5, j) + np.linspace(-0.12, 0.12, 5), v, s=16, color=COLORS[m], alpha=0.75,
                   edgecolor="white", linewidth=0.4, zorder=3)
        r = summary[(summary.split == sp) & (summary.model == m)].iloc[0]
        ax.errorbar(j + 0.3, r.accuracy_mean * 100,
                    yerr=[[100 * (r.accuracy_mean - r.accuracy_hierCI_lo)], [100 * (r.accuracy_hierCI_hi - r.accuracy_mean)]],
                    fmt="o", color=INK, ms=4, capsize=3, lw=1, zorder=4)
    ax.set_xticks(range(3), MODELS)
    ax.set_title(f"{SPLIT_LABEL[sp]} (n = {len(Y[sp])} test images)")
    ax.set_xlim(-0.5, 2.6)
axes[0].set_ylabel("Test accuracy (%)")
fig.text(0.5, -0.04, "Dots: individual seeds. Black: mean with 95 % hierarchical bootstrap CI (seeds × images).",
         ha="center", fontsize=7, color=INK2)
save(fig, "fig_accuracy_by_seed")

fig, ax = plt.subplots(figsize=(3.5, 2.7))
labels, y0 = [], 0
for sp, mk in zip(SPLITS, ["o", "s"]):
    sub = pairs[pairs.split == sp].reset_index(drop=True)
    for k, r in sub.iterrows():
        yy = k + (0.18 if sp == "original" else -0.18)
        ax.errorbar(r.acc_diff_mean * 100, yy, xerr=[[100 * (r.acc_diff_mean - r.acc_diff_hierCI_lo)],
                                                       [100 * (r.acc_diff_hierCI_hi - r.acc_diff_mean)]],
                    fmt=mk, color=INK if sp == "original" else INK2, mfc="white" if sp == "dedup" else INK,
                    ms=4, capsize=2.5, lw=1, label=SPLIT_LABEL[sp] if k == 0 else None)
        if sp == "original":
            labels.append(f"{r.model_a}\n− {r.model_b}")
ax.axvline(0, color=INK2, lw=0.8, ls="--")
ax.set_yticks(range(3), labels, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Accuracy difference (pp), 95 % hierarchical CI")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "fig_pairwise_differences")

# ------------------------------------------------ 9. training dynamics (mean curves, original split)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), constrained_layout=True)
for m in MODELS:
    for ax, ph in zip(axes, ["history_phase1", "history_phase2"]):
        curves = [R["original", m, s][ph]["val_loss"] for s in SEEDS]
        L = max(len(c) for c in curves)
        arr = np.full((5, L), np.nan)
        for i, c in enumerate(curves):
            arr[i, :len(c)] = c
        x = np.arange(1, L + 1)
        for c in curves:
            ax.plot(np.arange(1, len(c) + 1), c, color=COLORS[m], lw=0.5, alpha=0.35)
        ax.plot(x, np.nanmean(arr, 0), color=COLORS[m], lw=1.6, label=m)
axes[0].set_title("Phase 1: classification head only")
axes[1].set_title("Phase 2: top 30 % of backbone fine-tuned (BatchNorm frozen)")
for ax in axes:
    ax.set_xlabel("Epoch")
axes[0].set_ylabel("Validation loss")
axes[0].legend(frameon=False)
save(fig, "fig_training_curves")

# ------------------------------------------------ 10. explainability (Grad-CAM deletion / insertion)
xai_rows = []
for (sp, m, s), r in R.items():
    for rec in r["xai"]:
        xai_rows.append(dict(split=sp, model=m, seed=s, image=rec["relpath"], pred=rec["pred"], conf=rec["conf"],
                             deletion=rec["deletion_auc"], insertion=rec["insertion_auc"]))
xai = pd.DataFrame(xai_rows)
xai["ins_minus_del"] = xai.insertion - xai.deletion
lab = {sp: dict(zip(PATHS[sp], Y[sp])) for sp in SPLITS}
xai["correct"] = [int(p == lab[sp][i]) for sp, i, p in zip(xai.split, xai.image, xai.pred)]
xai_layers = {m: R["original", m, 42]["xai_protocol"]["gradcam_layer"] for m in MODELS}


def xai_hier_ci(sub, col):
    mat = sub.pivot(index="seed", columns="image", values=col).values
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        out[b] = mat[RNG.integers(0, 5, 5)][:, RNG.integers(0, mat.shape[1], mat.shape[1])].mean()
    return np.percentile(out, [2.5, 97.5])


xs = []
for sp in SPLITS:
    for m in MODELS:
        sub = xai[(xai.split == sp) & (xai.model == m)]
        row = dict(split=sp, model=m, gradcam_layer=xai_layers[m])
        for col in ["deletion", "insertion", "ins_minus_del"]:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_seed_sd"] = sub.groupby("seed")[col].mean().std(ddof=1)
            row[f"{col}_CI_lo"], row[f"{col}_CI_hi"] = xai_hier_ci(sub, col)
        per_img = sub.groupby("image")["ins_minus_del"]
        # seed stability: share of image-level variance that is between images (ICC(1)-style)
        msb = per_img.mean().var(ddof=1) * 5
        msw = per_img.var(ddof=1).mean()
        row["ins_minus_del_icc1"] = (msb - msw) / (msb + 4 * msw)
        row["spearman_conf_vs_insdel"] = stats.spearmanr(sub.conf, sub.ins_minus_del).statistic
        row["ins_minus_del_correct"] = sub[sub.correct == 1].ins_minus_del.mean()
        row["ins_minus_del_wrong"] = sub[sub.correct == 0].ins_minus_del.mean() if (sub.correct == 0).any() else np.nan
        xs.append(row)
xai_summary = pd.DataFrame(xs)
xai_summary.to_csv(TAB / "xai_summary.csv", index=False)
key["xai"] = xai_summary.round(4).to_dict(orient="records")

xp = []
for sp in SPLITS:
    img = {m: xai[(xai.split == sp) & (xai.model == m)].groupby("image").ins_minus_del.mean() for m in MODELS}
    for a, b in combinations(MODELS, 2):
        d = (img[a] - img[b]).values
        w = stats.wilcoxon(d)
        # matched-pairs rank-biserial correlation as effect size
        rk = stats.rankdata(np.abs(d))
        rbc = (rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum()
        xp.append(dict(split=sp, model_a=a, model_b=b, mean_diff=d.mean(), wilcoxon_p=w.pvalue,
                       rank_biserial=rbc))
xai_pairs = pd.DataFrame(xp)
xai_pairs["wilcoxon_p_holm"] = xai_pairs.groupby("split").wilcoxon_p.transform(holm)
xai_pairs.to_csv(TAB / "xai_pairwise.csv", index=False)
key["xai_pairwise"] = xai_pairs.round(6).to_dict(orient="records")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
for ax, col, title in zip(axes, ["deletion", "insertion", "ins_minus_del"],
                          ["Deletion AUC (lower = more faithful)", "Insertion AUC (higher = more faithful)",
                           "Insertion − deletion"]):
    pos, data, cols = [], [], []
    for j, m in enumerate(MODELS):
        for k, sp in enumerate(SPLITS):
            pos.append(j + (k - 0.5) * 0.36)
            data.append(xai[(xai.split == sp) & (xai.model == m)].groupby("image")[col].mean().values)
            cols.append(COLORS[m])
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1))
    for patch, c, k in zip(bp["boxes"], cols, [0, 1] * 3):
        patch.set_facecolor(c)
        patch.set_alpha(0.85 if k == 0 else 0.4)
        patch.set_edgecolor(INK2)
    ax.set_xticks(range(3), ["MobileNet\nV3Small", "Efficient\nNetB0", "Dense\nNet121"])
    ax.set_title(title, fontsize=8)
fig.text(0.5, -0.05, "Per-image means over 5 seeds. Solid: original split; light: de-duplicated split.",
         ha="center", fontsize=7, color=INK2)
save(fig, "fig_xai_faithfulness")

# ------------------------------------------------ 11. efficiency: latency, TFLite, Pareto
lat = pd.read_csv(RAW / "latency_benchmark.csv")
lat_s = lat.groupby(["version", "backbone"]).median_ms.agg(["median", "min", "max"]).reset_index()
lat_s.to_csv(TAB / "latency_keras_cpu.csv", index=False)
tfl = json.loads((RAW / "tflite_benchmark.json").read_text())
tfl_rows = [dict(model=m, variant=v, **tfl[m][v]) for m in MODELS for v in ("float32", "int8_dynamic")]
tflite = pd.DataFrame(tfl_rows)
tflite.to_csv(TAB / "tflite_benchmark.csv", index=False)
key["latency_keras_cpu_ms"] = {f"{r.version}/{r.backbone}": round(r.median, 2) for r in lat_s.itertuples()}
key["tflite"] = tfl
pkg = (RAW / "package_versions.txt").read_text()
key["environment"] = dict(line.split(": ", 1) for line in pkg.strip().splitlines())

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
for ax, sp in zip(axes, SPLITS):
    for m in MODELS:
        r = summary[(summary.split == sp) & (summary.model == m)].iloc[0]
        l_ = lat_s[(lat_s.version == sp) & (lat_s.backbone == m)]["median"].iloc[0]
        ax.errorbar(l_, r.accuracy_mean * 100,
                    yerr=[[100 * (r.accuracy_mean - r.accuracy_hierCI_lo)], [100 * (r.accuracy_hierCI_hi - r.accuracy_mean)]],
                    fmt="o", color=COLORS[m], ms=3.5 + r.params_million * 0.9, capsize=3, lw=1, alpha=0.9,
                    label=f"{m} ({r.params_million:.2f} M params)")
    ax.set_xscale("log")
    ax.set_xlabel("CPU latency per image (ms, log scale)")
    ax.set_title(SPLIT_LABEL[sp])
    ax.set_xlim(7, 250)
axes[0].set_ylabel("Test accuracy (%), mean and 95 % CI")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7.5)
save(fig, "fig_pareto")

# ------------------------------------------------ 12. seed stability and error consistency
churn_rows = []
for sp in SPLITS:
    for m in MODELS:
        P = PRED[sp, m]
        dis = [np.mean(P[i] != P[j]) for i, j in combinations(range(5), 2)]
        churn_rows.append(dict(split=sp, model=m, mean_pairwise_disagreement=np.mean(dis),
                               images_always_correct=int((CORR[sp, m].sum(0) == 5).sum()),
                               images_never_correct=int((CORR[sp, m].sum(0) == 0).sum()),
                               images_unstable=int(((CORR[sp, m].sum(0) > 0) & (CORR[sp, m].sum(0) < 5)).sum())))
churn = pd.DataFrame(churn_rows)
churn.to_csv(TAB / "seed_stability.csv", index=False)
hard = []
for sp in SPLITS:
    tot = sum(CORR[sp, m] for m in MODELS).sum(0)          # correct out of 15 (3 models x 5 seeds)
    for i in np.where(tot < 15)[0]:
        wrong = np.concatenate([PRED[sp, m][:, i][CORR[sp, m][:, i] == 0] for m in MODELS])
        hard.append(dict(split=sp, image=PATHS[sp][i], true=CLASS_LABELS[Y[sp][i]], correct_of_15=int(tot[i]),
                         most_common_wrong=CLASS_LABELS[np.bincount(wrong, minlength=K).argmax()]))
pd.DataFrame(hard).sort_values(["split", "correct_of_15"]).to_csv(TAB / "hard_images.csv", index=False)
key["seed_stability"] = churn.round(4).to_dict(orient="records")

# ------------------------------------------------ 13. how implementation defects changed the conclusions
hist_rows = []
v2 = json.loads((ROOT / "results" / "v2_repeated_seeds" / "raw" / "results_summary.json").read_text())
for m in MODELS:
    a = np.array([v2[m][str(s)]["test_accuracy"] for s in SEEDS])
    hist_rows.append(dict(protocol="v2: callback state shared across phases", split="original", model=m,
                          acc_mean=a.mean(), acc_sd=a.std(ddof=1)))
run1 = ROOT / "archive" / "results" / "v3_run1_bn_training_mode" / "runs"
for sp in SPLITS:
    for m in MODELS:
        a = np.array([json.loads((run1 / sp / f"{m}_seed{s}.json").read_text())["test_accuracy"] for s in SEEDS])
        hist_rows.append(dict(protocol="v3 run 1: BatchNorm in training mode during fine-tuning", split=sp,
                              model=m, acc_mean=a.mean(), acc_sd=a.std(ddof=1)))
for sp in SPLITS:
    for m in MODELS:
        a = per_run[(per_run.split == sp) & (per_run.model == m)].accuracy.values
        hist_rows.append(dict(protocol="v3 final: corrected protocol", split=sp, model=m,
                              acc_mean=a.mean(), acc_sd=a.std(ddof=1)))
protocol_hist = pd.DataFrame(hist_rows)
protocol_hist.to_csv(TAB / "protocol_history.csv", index=False)
key["protocol_history"] = protocol_hist.round(6).to_dict(orient="records")

fig, ax = plt.subplots(figsize=(3.5, 2.7))
sub = protocol_hist[protocol_hist.split == "original"]
prots = list(dict.fromkeys(sub.protocol))
short = ["v2\nshared\ncallbacks", "v3 run 1\nBatchNorm in\ntraining mode", "v3 final\ncorrected"]
for j, m in enumerate(MODELS):
    s_ = sub[sub.model == m].set_index("protocol").loc[prots]
    x = np.arange(3) + (j - 1) * 0.12
    ax.errorbar(x, s_.acc_mean * 100, yerr=s_.acc_sd * 100, fmt="-o", color=COLORS[m], ms=4, capsize=2.5,
                lw=1.2, label=m)
ax.set_xticks(range(3), short, fontsize=7)
ax.set_ylabel("Test accuracy, original split (%)")
ax.legend(frameon=False, fontsize=7)
save(fig, "fig_protocol_history")

(OUT / "key_numbers.json").write_text(json.dumps(key, indent=1, default=float))
print("wrote", OUT)
print(summary[["split", "model", "accuracy_mean", "accuracy_sd", "accuracy_hierCI_lo", "accuracy_hierCI_hi",
               "macro_f1_mean", "mcc_mean", "phase2_selected"]].round(4).to_string(index=False))
print(pairs[["split", "model_a", "model_b", "acc_diff_mean", "acc_diff_hierCI_lo", "acc_diff_hierCI_hi",
             "image_perm_p_holm", "mcnemar_fisher_p_holm"]].round(4).to_string(index=False))
print(gee.round(4).to_string(index=False))
print(split_cmp.round(4).to_string(index=False))
print(gee_split.round(4).to_string(index=False))
print(xai_summary[["split", "model", "deletion_mean", "insertion_mean", "ins_minus_del_mean",
                   "ins_minus_del_CI_lo", "ins_minus_del_CI_hi", "ins_minus_del_icc1"]].round(3).to_string(index=False))
print(xai_pairs.round(4).to_string(index=False))
print(churn.to_string(index=False))
print(protocol_hist.round(4).to_string(index=False))
print(json.dumps({k: key[k] for k in ("duplicates", "original_test_images_with_cross_split_duplicate",
                                      "acc_on_duplicated_vs_clean_test_images")}, indent=1, default=float))
