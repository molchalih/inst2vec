"""Paper-facing summary table for all embedding cases."""

from __future__ import annotations

from sqlalchemy.orm import Session

from modules.cluster_results import (
    DEFAULT_CASES,
    list_case_rows,
    summarize_case_rows,
)

__all__ = (
    "DEFAULT_CASES",
    "summarize_all_to_markdown",
    "summarize_case_for_markdown",
    "summarize_to_markdown",
)

TABLE_ROWS: tuple[tuple[str, str], ...] = (
    (r"$n_{\mathrm{runs}}$", "n_runs"),
    (r"$n_{\mathrm{filtered}}$", "n_filtered"),
    (r"$\mu_{\mathrm{dbcv}}$", "dbcv_mean"),
    (r"$\sigma_{\mathrm{dbcv}}$", "dbcv_std"),
    (r"$\mu_{\mathrm{sil}}$", "silhouette_mean"),
    (r"$\tilde{k}$", "k_median"),
    (r"$\mu_{\mathrm{noise}}$ (%)", "noise_pct_mean"),
)


def summarize_case_for_markdown(session, case: str) -> dict[str, str]:
    rows = list_case_rows(session, case)
    return summarize_case_rows(rows)


def summarize_to_markdown(eng, case: str) -> str:
    return summarize_all_to_markdown(eng, cases=(case,))


def summarize_all_to_markdown(
    eng,
    *,
    cases: tuple[str, ...] = DEFAULT_CASES,
) -> str:
    if not cases:
        raise ValueError("cases must contain at least one embedding case")

    with Session(eng) as session:
        summaries = {case: summarize_case_for_markdown(session, case) for case in cases}
    col_header = " | ".join(cases)
    align_row = "|".join(["---:"] * len(cases))
    table_lines = [
        f"| Metric | {col_header} |",
        f"|---|{align_row}|",
    ]
    for label, key in TABLE_ROWS:
        values = [summaries[case][key] for case in cases]
        table_lines.append(f"| {label} | {' | '.join(values)} |")
    table_lines.extend(
        [
            "",
            ": Selected clustering metrics across embedding cases.",
        ]
    )
    return "\n".join(table_lines)
