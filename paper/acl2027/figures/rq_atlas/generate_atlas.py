"""Render the independent RQ atlas on mgt01 and assemble its PDF/contact sheet."""

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import textwrap

import matplotlib.pyplot as plt

from atlas_common import load_evidence

MODULES = ["plot_rq1_rq2", "plot_rq3", "plot_rq4_rq5", "plot_training_resources"]


def assemble(directory, records):
    directory = Path(directory)
    records = list(records)
    subprocess.run(["pdfunite", *[str(directory/"plots"/r["pdf"]) for r in records],
                    str(directory/"rq1_rq5_complete_atlas.pdf")], check=True)
    columns = 4
    rows = (len(records)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, rows*1.8), squeeze=False)
    fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.93, wspace=.08, hspace=.26)
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index < len(records):
            record = records[index]
            ax.imshow(plt.imread(directory/"plots"/record["png"]))
            ax.set_title(textwrap.fill(f"{index+1:02d}  {record['title']}", 47), fontsize=8, pad=4)
    fig.suptitle("RQ1–RQ5  |  Complete visual atlas", fontsize=15, fontweight="bold", x=.02, ha="left")
    fig.savefig(directory/"contact_sheet.png", dpi=160)
    plt.close(fig)
    manifest = {"backup_tag": "backup/acl-rq-atlas-20261104", "figure_count": len(records),
                "source_files": ["paper/acl2027/evidence/data/metrics.json", "paper/acl2027/evidence/data/comparisons.json"],
                "figures": records}
    (directory/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(f"Assembled {len(records)} figures: {directory / 'rq1_rq5_complete_atlas.pdf'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2]/"evidence"/"data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    metrics, comparisons = load_evidence(args.data_dir)
    records = []
    for name in MODULES:
        records.extend(importlib.import_module(name).build(metrics, comparisons, args.output_dir/"plots"))
    assemble(args.output_dir, records)


if __name__ == "__main__":
    main()
