"""
Re-labels the vis-ss training CSVs by applying the ambiguous-center rule:
the CENTER panel (panel 2 in singles, panels 2/7 in doubles) of a triple/quad
matching one of the patterns 11100, 01110, 00111, 10101, 10110, 01011 is
relabeled to 'e' (either-foot) when:
  (a) the previous line had a downpress on the same center (jack on center), OR
  (b) the previous line had a downpress on a *different* panel annotated l/r
      (so prior body orientation is set).

INPUT : artifacts/manual-chartstructs/visss-120524/*.csv
OUTPUT: artifacts/manual-chartstructs/visss-120524-eaware/*.csv

After running this, retrain pointing --manual_chart_struct_folder at the
*-eaware folder and the model will learn the new rule from data.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from loguru import logger

from piu_annotate.formats import notelines

SRC_DIR = "/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/manual-chartstructs/visss-120524"
DST_DIR = "/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/manual-chartstructs/visss-120524-eaware"


def relabel_chart(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """ Returns (new_df, n_relabeled_rows). Walks rows in order so the
        previous line context is consistent with the current annotation.
    """
    df = df.copy()
    n_changed = 0
    prev_line: str | None = None
    prev_annot: str | None = None
    new_col = []
    for _, row in df.iterrows():
        line_ah = str(row["Line with active holds"])
        annot = str(row["Limb annotation"]) if not pd.isna(row["Limb annotation"]) else ""
        new_annot = notelines.relabel_with_ambiguous_e(
            line_ah, annot, prev_line, prev_annot
        )
        if new_annot != annot:
            n_changed += 1
        new_col.append(new_annot)
        prev_line = line_ah
        prev_annot = new_annot
    df["Limb annotation"] = new_col
    return df, n_changed


def main():
    src = Path(SRC_DIR)
    dst = Path(DST_DIR)
    dst.mkdir(parents=True, exist_ok=True)

    csvs = sorted(src.glob("*.csv"))
    logger.info(f"Found {len(csvs)} CSVs in {src}")

    total_changed_rows = 0
    total_files = 0
    files_with_changes = 0
    for csv in tqdm(csvs):
        try:
            df = pd.read_csv(csv, dtype={"Limb annotation": str})
        except Exception as e:
            logger.warning(f"Skip {csv.name}: {e}")
            continue
        df["Limb annotation"] = df["Limb annotation"].fillna("")
        new_df, n_changed = relabel_chart(df)
        out = dst / csv.name
        new_df.to_csv(out, index=False)
        total_files += 1
        if n_changed:
            files_with_changes += 1
            total_changed_rows += n_changed

    logger.success(
        f"Done. {total_files} files written -> {dst}. "
        f"{files_with_changes} files modified, {total_changed_rows} rows relabeled."
    )

    # Also copy non-CSV companion files (yaml indexes etc.) if they exist.
    for f in src.iterdir():
        if not f.is_file() or f.suffix == ".csv":
            continue
        shutil.copy2(f, dst / f.name)


if __name__ == "__main__":
    main()
