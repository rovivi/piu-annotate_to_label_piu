from __future__ import annotations
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from loguru import logger

from piu_annotate.formats import notelines
from piu_annotate.difficulty.travel import pos as pos_map


@dataclass
class ArrowDataPoint:
    """ Datapoint representing a single arrow.
        A line can have multiple arrows.
        This should not use any limb information for any arrow.
    """
    arrow_pos: int
    arrow_symbol: str
    line_with_active_holds: str
    active_hold_idxs: list[int]
    prior_line_only_releases_hold_on_this_arrow: bool
    time_since_last_same_arrow_use: float
    time_since_prev_downpress: float
    num_downpress_in_line: int
    line_is_bracketable: bool
    line_repeats_previous_downpress_line: bool
    line_repeats_next_downpress_line: bool
    singles_or_doubles: str
    prev_pc_idxs: list[int | None]
    next_line_only_releases_hold_on_this_arrow: bool
    x: float = 0.0
    y: float = 0.0
    # v9 structural features
    prev_line_was_bracket: bool = False
    next_line_is_bracketable: bool = False
    hold_count_in_line: int = 0
    time_to_next_downpress: float = -1.0
    is_jack: bool = False
    n_same_panel_streak: int = 0

    def to_array_categorical(self) -> NDArray:
        """ Featurize, using int for categorical features """
        assert self.singles_or_doubles in ['singles', 'doubles']
        line_ft = [int(c) for c in self.line_with_active_holds]
        fts = [
            self.arrow_pos,
            int(self.arrow_symbol),
            int(len(self.active_hold_idxs) > 0),
            int(self.prior_line_only_releases_hold_on_this_arrow),
            self.time_since_last_same_arrow_use,
            self.time_since_prev_downpress, 
            self.num_downpress_in_line,
            int(self.line_is_bracketable),
            int(self.line_repeats_previous_downpress_line),
            int(self.line_repeats_next_downpress_line),
            self.x,
            self.y,
            # v9
            int(self.prev_line_was_bracket),
            int(self.next_line_is_bracketable),
            self.hold_count_in_line,
            self.time_to_next_downpress,
            int(self.is_jack),
            self.n_same_panel_streak,
        ]
        return np.concatenate([np.array(fts), line_ft])

    def get_feature_names_categorical(self) -> list[str]:
        """ Must be aligned with categorical array """
        sord = self.singles_or_doubles
        length = 5 if sord == 'singles' else 10
        line_ft_names = [f'cat.line_pos{idx}' for idx in range(length)]
        ft_names = [
            'cat.arrow_pos',
            'cat.arrow_symbol',
            'has_active_hold',
            'prior_line_only_releases_hold_on_this_arrow',
            'time_since_last_same_arrow_use',
            'time_since_prev_downpress',
            'num_downpress_in_line',
            'line_is_bracketable',
            'line_repeats_previous_downpress_line',
            'line_repeats_next_downpress_line',
            'x',
            'y',
            # v9
            'prev_line_was_bracket',
            'next_line_is_bracketable',
            'hold_count_in_line',
            'time_to_next_downpress',
            'is_jack',
            'n_same_panel_streak',
        ] + line_ft_names
        assert len(ft_names) == len(self.to_array_categorical())
        return ft_names


@dataclass
class LimbLabel:
    limb: int   # 0 for left, 1 for right, 2 for either (e)

    @staticmethod
    def from_limb_annot(annot: str):
        mapper = {'l': 0, 'r': 1, 'h': 0, 'e': 2}
        return LimbLabel(limb=mapper[annot])

    def to_array(self) -> NDArray:
        return np.array(self.limb)
    
