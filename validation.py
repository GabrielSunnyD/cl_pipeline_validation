import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

SAMPLE_PATH = "gabriel_validation_sample.csv"
LABELS_PATH = "gabriel_validation_labels.csv"
ERRORS_PATH = "gabriel_validation_errors.csv"
FIGURES_DIR = "figures"

PAIRS = [
    ("gt_judge_name", "p_judge_name"),
    ("gt_signer_role", "p_signer_role"),
    ("gt_disposition", "p_disposition"),
    ("gt_case_type", "p_case_type"),
]

PAIR_LABELS = {
    "gt_judge_name": "Judge Name",
    "gt_signer_role": "Signer Role",
    "gt_disposition": "Disposition",
    "gt_case_type": "Case Type",
}

# Validated categorical palette (blue/orange/aqua/yellow/...) - fixed hue
# order clears the CVD-adjacent and normal-vision-floor checks for bar charts
# (validated via the dataviz skill's validate_palette.js). Always slice from
# the front; never reorder or cycle.
CATEGORICAL_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]


GT_COLS = [gt for gt, _ in PAIRS]


def load_data():
    sample = pd.read_csv(SAMPLE_PATH, dtype=str).fillna("")
    labels = pd.read_csv(LABELS_PATH, dtype=str).fillna("")
    df = sample.merge(labels, on="opinion_id")

    # Rows with no ground truth at all (OCR garbage / no opinion available)
    # can't inform any accuracy computation - confirmed via gt_notes and
    # excluded here so every downstream figure is computed on the same base.
    no_gt = (df[GT_COLS] == "").all(axis=1)
    if no_gt.any():
        print(f"Dropping {no_gt.sum()} rows with no ground truth (OCR garbage / no opinion):")
        for _, row in df[no_gt].iterrows():
            print(f"    opinion_id={row['opinion_id']}: {row['gt_notes']}")
        df = df[~no_gt].reset_index(drop=True)

    return df


def is_match(gt_col, gt_val, pred_val):
    """Row-level match test. Disposition predictions are comma-separated sets
    (e.g. "granted,denied"); we count agreement if the single gt value appears
    anywhere in that set, rather than requiring an exact string match."""
    if gt_col == "gt_disposition":
        if not pred_val:
            return False
        return gt_val in {p.strip() for p in pred_val.split(",")}
    return gt_val == pred_val


def match_series(df, gt_col, pred_col):
    return df.apply(
        lambda r: is_match(gt_col, r[gt_col], r[pred_col]), axis=1
    ).rename("match")


def agreement_rate(df, gt_col, pred_col):
    return match_series(df, gt_col, pred_col).mean()


def print_section(title):
    print(f"\n{'='*60}")
    print(title)
    print('='*60)


def breakdown_agreement(df, gt_col, pred_col, by_col):
    match = match_series(df, gt_col, pred_col)
    return df[[by_col]].join(match).groupby(by_col)["match"].mean()


def confusion_matrix(df, gt_col, pred_col):
    all_labels = sorted(set(df[gt_col].unique()) | set(df[pred_col].unique()))
    return pd.crosstab(
        df[gt_col],
        df[pred_col],
        rownames=[gt_col],
        colnames=[pred_col],
    ).reindex(index=all_labels, columns=all_labels, fill_value=0)


def confusion_matrix_disposition(df, gt_col, pred_col):
    """Confusion matrix that respects the disposition substring-match rule.

    The pipeline emits comma-separated disposition sets. Collapsing any row
    where the gt value is found in that set onto the diagonal (rather than
    plotting the raw multi-value string as its own row/column) is what keeps
    this matrix legible; only genuine mismatches keep the raw pipeline string
    so you can still see exactly what it said.
    """
    def matched_label(gt_val, pred_val):
        if not pred_val:
            return "(none)"
        preds = {p.strip() for p in pred_val.split(",")}
        return gt_val if gt_val in preds else pred_val

    matched = df.apply(lambda r: matched_label(r[gt_col], r[pred_col]), axis=1)
    row_labels = sorted(df[gt_col].unique())
    col_labels = sorted(set(matched.unique()) | set(row_labels))
    return pd.crosstab(
        df[gt_col], matched, rownames=[gt_col], colnames=[pred_col]
    ).reindex(index=row_labels, columns=col_labels, fill_value=0)


def save_fig(name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Figure helpers ────────────────────────────────────────────────────────────

def render_table(col_headers, row_headers, cell_values, title, fname,
                 col_widths=None):
    """Render a plain matplotlib table and save it."""
    n_rows = len(row_headers)
    n_cols = len(col_headers)
    fig_h = max(1.2, 0.45 * n_rows + 0.8)
    fig_w = max(4, sum(col_widths) if col_widths else n_cols * 1.8 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_values,
        rowLabels=row_headers,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    if col_widths:
        for (row, col), cell in table.get_celld().items():
            if col == -1:
                cell.set_width(col_widths[0] if col_widths else 0.2)
            elif col < len(col_widths) - 1:
                cell.set_width(col_widths[col + 1])
    # Style header row and row-label column
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EEF2F8" if row % 2 == 0 else "white")
        cell.set_edgecolor("#cccccc")

    ax.set_title(title, pad=12, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fname)


def fig_overall_agreement(rates):
    """Horizontal bar chart of overall agreement rates (section 1)."""
    labels = [PAIR_LABELS[gt] for gt, _ in PAIRS]
    values = [rates[gt] * 100 for gt, _ in PAIRS]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(labels, values, color="#4C72B0", height=0.5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_xlabel("Agreement rate")
    ax.set_title("Overall Pipeline Agreement Rates")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig("1_overall_agreement.png")


def fig_overall_agreement_table(rates, n):
    """Table: Field | Agreement Rate | Correct / Total (section 1)."""
    row_headers = [PAIR_LABELS[gt] for gt, _ in PAIRS]
    col_headers = ["Agreement Rate", "Correct / Total"]
    cell_values = [
        [f"{rates[gt]:.1%}", f"{int(rates[gt] * n)} / {n}"]
        for gt, _ in PAIRS
    ]
    render_table(
        col_headers, row_headers, cell_values,
        title="Overall Pipeline Agreement Rates",
        fname="1_overall_agreement_table.png",
        col_widths=[1.6, 1.4, 1.4],
    )


def fig_breakdown_agreement(breakdown_data, by_col, pairs=PAIRS):
    """Grouped bar chart of agreement rates broken down by a column (section 2)."""
    field_labels = [PAIR_LABELS[gt] for gt, _ in pairs]
    groups = sorted({g for rates in breakdown_data.values() for g in rates.index})
    n_groups = len(groups)
    n_fields = len(pairs)

    x = range(n_fields)
    width = 0.8 / n_groups
    palette = CATEGORICAL_PALETTE[:n_groups]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, group in enumerate(groups):
        offsets = [j + (i - n_groups / 2 + 0.5) * width for j in x]
        values = [breakdown_data[gt].get(group, 0) * 100 for gt, _ in pairs]
        ax.bar(offsets, values, width=width * 0.9, label=group, color=palette[i])

    ax.set_xticks(list(x))
    ax.set_xticklabels(field_labels)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_ylabel("Agreement rate")
    ax.set_title(f"Agreement Rates by {by_col}")
    ax.legend(title=by_col, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    save_fig(f"2_agreement_by_{by_col}.png")


def fig_breakdown_agreement_table(breakdown_data, counts, by_col, pairs=PAIRS):
    """Table: Field rows × group columns with 'rate (k/n)' cells (section 2)."""
    groups = sorted({g for bd in breakdown_data.values() for g in bd.index})
    row_headers = [PAIR_LABELS[gt] for gt, _ in pairs]
    col_headers = groups
    cell_values = []
    for gt, _ in pairs:
        row = []
        for g in groups:
            rate = breakdown_data[gt].get(g, float("nan"))
            n = counts.get(g, 0)
            row.append(f"{rate:.1%}  ({int(rate * n)}/{n})" if pd.notna(rate) else "—")
        cell_values.append(row)
    render_table(
        col_headers, row_headers, cell_values,
        title=f"Agreement Rates by {by_col}",
        fname=f"2_agreement_by_{by_col}_table.png",
        col_widths=[1.6] + [1.6] * len(groups),
    )


def fig_signer_role_accuracy(df):
    """Bar chart + table: gt_signer_role vs p_signer_role accuracy, restricted
    to rows where the pipeline actually made a claim (district/magistrate).
    Excludes "unclear" rows entirely - those are covered by the recovery
    analysis below, since the pipeline made no claim to be right or wrong
    about there."""
    claimed = df[df["p_signer_role"].isin(["district", "magistrate"])]
    groups = ["district", "magistrate"]
    rates, counts = {}, {}
    for g in groups:
        sub = claimed[claimed["p_signer_role"] == g]
        counts[g] = len(sub)
        rates[g] = (sub["gt_signer_role"] == sub["p_signer_role"]).mean() if len(sub) else float("nan")
    overall_rate = (claimed["gt_signer_role"] == claimed["p_signer_role"]).mean()
    overall_n = len(claimed)

    row_labels = groups + ["Overall (claimed)"]
    values = [rates[g] * 100 for g in groups] + [overall_rate * 100]
    ns = [counts[g] for g in groups] + [overall_n]
    correct = [int(rates[g] * counts[g]) for g in groups] + [int(overall_rate * overall_n)]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(row_labels, values, color="#4C72B0", height=0.5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_xlabel("Accuracy")
    ax.set_title("Signer Role Accuracy (rows where pipeline made a claim)")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig("2_signer_role_accuracy.png")

    render_table(
        ["Accuracy", "Correct / Total"],
        row_labels,
        [[f"{v:.1f}%", f"{c} / {n}"] for v, c, n in zip(values, correct, ns)],
        title="Signer Role Accuracy (rows where pipeline made a claim)",
        fname="2_signer_role_accuracy_table.png",
        col_widths=[1.8, 1.6, 1.6],
    )
    return rates, overall_rate, overall_n


def fig_signer_role_recovery(df):
    """Bar chart + table: for rows the pipeline left "unclear", what fraction
    did ground-truth review manage to resolve to an actual role, and what did
    those rows turn out to be. This is a recovery rate, not an accuracy rate -
    the pipeline made no claim on these rows, so there's nothing for it to
    have gotten "wrong"."""
    unclear = df[df["p_signer_role"] == "unclear"]
    n = len(unclear)
    resolved = unclear[unclear["gt_signer_role"] != ""]
    recovery_rate = len(resolved) / n if n else float("nan")

    composition = resolved["gt_signer_role"].value_counts()
    row_labels = list(composition.index)
    values = [composition[r] / n * 100 for r in row_labels]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * len(row_labels) + 1)))
    bars = ax.barh(row_labels, values, color="#55A868", height=0.5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_xlabel("Share of pipeline 'unclear' rows")
    ax.set_title(
        f"Signer Role Recovery on Pipeline 'Unclear' Rows "
        f"(recovered {len(resolved)}/{n} = {recovery_rate:.1%})"
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig("2_signer_role_recovery.png")

    row_headers = row_labels + ["Recovered (any role)"]
    cell_values = [
        [f"{composition[r]}", f"{composition[r] / n:.1%}"] for r in row_labels
    ] + [[f"{len(resolved)}", f"{recovery_rate:.1%}"]]
    render_table(
        ["Count", "Share of Unclear"],
        row_headers,
        cell_values,
        title=f"Signer Role Recovery on Pipeline 'Unclear' Rows (n={n})",
        fname="2_signer_role_recovery_table.png",
        col_widths=[2.2, 1.4, 1.6],
    )
    return recovery_rate, n, composition


def fig_confusion_matrix(cm, title, fname):
    """Annotated heatmap for a confusion matrix (section 3)."""
    max_label_len = max((len(str(c)) for c in cm.columns), default=0)
    col_width = 0.6 if max_label_len <= 12 else 0.85
    fig, ax = plt.subplots(figsize=(max(6, len(cm.columns) * col_width + 2),
                                    max(5, len(cm.index) * 0.5 + 2.5)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", linewidths=0.4,
        linecolor="#cccccc", ax=ax, cbar_kws={"shrink": 0.7},
    )
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    save_fig(fname)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    # 1. Overall agreement rates
    print_section("1. Overall Agreement Rates")
    rates = {}
    for gt_col, pred_col in PAIRS:
        rate = agreement_rate(df, gt_col, pred_col)
        rates[gt_col] = rate
        print(f"  {gt_col} vs {pred_col}: {rate:.1%} ({int(rate * len(df))}/{len(df)})")
    fig_overall_agreement(rates)
    fig_overall_agreement_table(rates, len(df))

    # 2. Agreement rates broken down by era and p_signer_role
    for by_col in ("era", "p_signer_role"):
        print_section(f"2. Agreement Rates by {by_col}")
        # gt_signer_role vs p_signer_role broken down *by p_signer_role* is
        # tautological in the "unclear" cell (the pipeline made no claim
        # there, so it can't "agree") - that split is handled separately
        # below by fig_signer_role_accuracy / fig_signer_role_recovery.
        pairs = PAIRS
        if by_col == "p_signer_role":
            pairs = [(gt, pred) for gt, pred in PAIRS if gt != "gt_signer_role"]
        breakdown_data = {}
        counts = df.groupby(by_col).size()
        for gt_col, pred_col in pairs:
            print(f"\n  {gt_col} vs {pred_col}:")
            breakdown = breakdown_agreement(df, gt_col, pred_col, by_col)
            breakdown_data[gt_col] = breakdown
            for group in breakdown.index:
                rate = breakdown[group]
                n = counts[group]
                print(f"    {group:20s}: {rate:.1%} ({int(rate * n)}/{n})")
        fig_breakdown_agreement(breakdown_data, by_col, pairs)
        fig_breakdown_agreement_table(breakdown_data, counts, by_col, pairs)

    print_section("2. Signer Role Accuracy vs Recovery")
    rates, overall_rate, overall_n = fig_signer_role_accuracy(df)
    for g, r in rates.items():
        print(f"  {g:12s}: {r:.1%}")
    print(f"  {'Overall (claimed)':12s}: {overall_rate:.1%} ({overall_n} rows)")
    recovery_rate, unclear_n, composition = fig_signer_role_recovery(df)
    print(f"\n  Recovery rate on 'unclear' rows: {recovery_rate:.1%} ({unclear_n} rows)")
    print(composition.to_string())

    # 3. Confusion matrices
    print_section("3. Confusion Matrix: gt_disposition × p_disposition")
    cm_disp = confusion_matrix_disposition(df, "gt_disposition", "p_disposition")
    print(cm_disp.to_string())
    fig_confusion_matrix(
        cm_disp,
        "Confusion Matrix: gt_disposition × p_disposition (substring match)",
        "3_confusion_disposition.png",
    )

    print_section("3. Confusion Matrix: gt_case_type × p_case_type")
    cm_case = confusion_matrix(df, "gt_case_type", "p_case_type")
    print(cm_case.to_string())
    fig_confusion_matrix(
        cm_case,
        "Confusion Matrix: gt_case_type × p_case_type",
        "3_confusion_case_type.png",
    )

    # 4. Errors CSV (only written if it doesn't already exist)
    print_section("4. Errors CSV")
    if os.path.exists(ERRORS_PATH):
        print(f"  {ERRORS_PATH} already exists, skipping.")
    else:
        error_mask = pd.Series(False, index=df.index)
        for gt_col, pred_col in PAIRS:
            error_mask |= ~match_series(df, gt_col, pred_col)

        errors = df[error_mask].copy()
        error_cols = ["opinion_id", "era", "p_signer_role"]
        for gt_col, pred_col in PAIRS:
            error_cols += [gt_col, pred_col]
        errors[error_cols].to_csv(ERRORS_PATH, index=False)
        print(f"  {len(errors)} rows with at least one mismatch written to {ERRORS_PATH}")


if __name__ == "__main__":
    main()
