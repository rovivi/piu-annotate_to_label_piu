import pytest
from piu_annotate.formats.mirror import (
    mirror_line,
    mirror_limb_annot,
    mirror_chartstruct,
    PANEL_MIRROR_SINGLES,
    PANEL_MIRROR_DOUBLES,
)


def test_mirror_line_singles_symmetric():
    assert mirror_line('10001', 'singles') == '10001'


def test_mirror_line_singles():
    assert mirror_line('11000', 'singles') == '00011'


def test_mirror_limb_annot_llr():
    assert mirror_limb_annot('llr') == 'lrr'


def test_mirror_limb_annot_le():
    assert mirror_limb_annot('le') == 'er'


def test_mirror_chartstruct_roundtrip():
    from piu_annotate.formats.chart import ChartStruct
    import pandas as pd

    df = pd.DataFrame([
        {
            'Beat': 0.0,
            'Time': 0.0,
            'Line': '`11000',
            'Line with active holds': '`11000',
            'Limb annotation': 'll',
        },
        {
            'Beat': 1.0,
            'Time': 1.0,
            'Line': '`00101',
            'Line with active holds': '`00101',
            'Limb annotation': 'lr',
        },
    ])
    cs = ChartStruct(df)
    cs_mirrored = mirror_chartstruct(mirror_chartstruct(cs))

    original_csv = cs.to_csv_string() if hasattr(cs, 'to_csv_string') else None
    mirrored_csv = cs_mirrored.to_csv_string() if hasattr(cs_mirrored, 'to_csv_string') else None

    if original_csv is None or mirrored_csv is None:
        import io
        buf1 = io.StringIO()
        buf2 = io.StringIO()
        cs.df.to_csv(buf1, index=False)
        cs_mirrored.df.to_csv(buf2, index=False)
        assert buf1.getvalue() == buf2.getvalue()
    else:
        assert original_csv == mirrored_csv


def test_panel_mirror_constants():
    assert PANEL_MIRROR_SINGLES == {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    assert PANEL_MIRROR_DOUBLES == {0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}
