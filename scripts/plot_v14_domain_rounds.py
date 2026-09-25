#!/usr/bin/env python3
"""Plot round-wise V14 P0/P3/P5 success rates for four agent domains."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DOMAIN_AGENTS = {
    "Technology": {
        "aerospace_engineer_agent",
        "autonomous_driving_agent",
        "system_admin_agent",
    },
    "Education": {"academic_search_agent", "education_consultant_agent"},
    "Healthcare": {"medical_advisor_agent", "psychological_counselor_agent"},
    "Business": {
        "ecommerce_manager_agent",
        "financial_analyst_agent",
        "legal_consultant_agent",
    },
}

SERIES = {
    "P0": ("benign_success_count", "#0072B2", "o"),
    "P3": ("persistent3_success_count", "#D55E00", "s"),
    "P5": ("persistent5_success_count", "#009E73", "^"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_to_domain = {
        agent: domain for domain, agents in DOMAIN_AGENTS.items() for agent in agents
    }
    totals: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    row_counts: dict[tuple[str, int], int] = defaultdict(int)

    with args.input.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            agent = row["agent"]
            if agent not in agent_to_domain:
                raise ValueError(f"Unmapped agent: {agent}")
            domain = agent_to_domain[agent]
            round_number = int(row["round"])
            key = (domain, round_number)
            row_counts[key] += 1
            for column, _, _ in SERIES.values():
                totals[key][column] += int(row[column])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for domain, agents in DOMAIN_AGENTS.items():
        rounds = list(range(1, 11))
        expected_rows = len(agents)
        for round_number in rounds:
            actual_rows = row_counts[(domain, round_number)]
            if actual_rows != expected_rows:
                raise ValueError(
                    f"{domain} round {round_number}: expected {expected_rows} rows, "
                    f"found {actual_rows}"
                )

        figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
        denominator = 200 * expected_rows
        for axis, (label, (column, color, marker)) in zip(axes, SERIES.items()):
            values = [
                100 * totals[(domain, round_number)][column] / denominator
                for round_number in rounds
            ]
            axis.plot(
                rounds,
                values,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=5.5,
            )
            span = max(values) - min(values)
            padding = max(1.0, span * 0.2)
            axis.set_ylim(max(0.0, min(values) - padding), min(100.0, max(values) + padding))
            axis.set_ylabel(f"{label} (%)", fontsize=11)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.grid(axis="y", linestyle="--", alpha=0.35)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        axes[0].set_title(domain, fontsize=14, weight="bold")
        axes[-1].set_xlabel("Round", fontsize=12)
        axes[-1].set_xticks(rounds)
        figure.tight_layout()

        filename = f"v14_{domain.lower()}_round_p0_p3_p5.png"
        figure.savefig(args.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
