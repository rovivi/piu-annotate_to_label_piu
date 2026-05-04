from __future__ import annotations
from piu_annotate.formats.chart import ChartStruct


PANEL_MIRROR_SINGLES = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
PANEL_MIRROR_DOUBLES = {0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}


def mirror_line(line: str, sd: str) -> str:
    if sd == 'singles':
        mapping = PANEL_MIRROR_SINGLES
        line_len = 5
    elif sd == 'doubles':
        mapping = PANEL_MIRROR_DOUBLES
        line_len = 10
    else:
        raise ValueError(f'Unknown sd: {sd}')

    line = line.replace('`', '')
    if len(line) != line_len:
        raise ValueError(f'Expected {line_len} chars, got {len(line)}')

    mirrored = ['0'] * line_len
    for old_pos, new_pos in mapping.items():
        mirrored[new_pos] = line[old_pos]
    return '`' + ''.join(mirrored)


def mirror_limb_annot(annot: str) -> str:
    trans = {'l': 'r', 'r': 'l', 'e': 'e', 'h': 'h', '?': '?'}
    return ''.join(trans[c] for c in reversed(annot))


def mirror_chartstruct(cs: ChartStruct) -> ChartStruct:
    import pandas as pd

    sd = cs.singles_or_doubles()
    new_rows = []
    for _, row in cs.df.iterrows():
        line = row['Line']
        line_ah = row['Line with active holds']
        limb_annot = row['Limb annotation']

        new_line = mirror_line(line, sd)
        new_line_ah = mirror_line(line_ah, sd)
        new_limb_annot = mirror_limb_annot(limb_annot) if limb_annot else ''

        new_rows.append({
            'Beat': row['Beat'],
            'Time': row['Time'],
            'Line': new_line,
            'Line with active holds': new_line_ah,
            'Limb annotation': new_limb_annot,
        })

    new_df = pd.DataFrame(new_rows)
    if 'Metadata' in cs.df.columns:
        new_df['Metadata'] = cs.df['Metadata'].tolist()

    new_cs = ChartStruct(new_df, source_file=cs.source_file)
    return new_cs
