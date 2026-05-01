import pytest
from piu_annotate.formats.chart import ChartStruct, is_active_symbol, ArrowCoordinate


class TestChartStruct:
    def test_is_active_symbol(self):
        assert is_active_symbol('1') == True
        assert is_active_symbol('2') == True
        assert is_active_symbol('3') == True
        assert is_active_symbol('4') == True
        assert is_active_symbol('0') == False

    def test_arrow_coordinate_hash(self):
        ac = ArrowCoordinate(row_idx=1, arrow_pos=2, limb_idx=0, is_downpress=True, line_with_active_holds='10000')
        assert hash(ac) == hash((1, 2, 0, True))


class TestLimbAnnotations:
    def test_init_limb_annotations(self):
        pass

    def test_add_limb_annotations(self):
        pass