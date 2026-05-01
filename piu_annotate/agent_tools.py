"""Agente Tools para piu-annotate

Proporciona utilities para que agentes IA interactuen con el proyecto.
"""

from dataclasses import dataclass
from typing import Optional
import os

from piu_annotate.formats.sscfile import SongSSC
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict as predict_limbs
from piu_annotate.segment.segment import Segmenter
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.difficulty.models import DifficultySegmentModelPredictor
from piu_annotate.formats.jsplot import ChartJsStruct


@dataclass
class ChartAnalysisResult:
    chartstruct: ChartStruct
    sections: list
    pred_limbs: list
    difficulty: Optional[float] = None


class PiuAnnotateAgent:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self._model_suite = None
        self._difficulty_predictor = None
        self._segmenter = None

    @property
    def model_suite(self) -> ModelSuite:
        if self._model_suite is None:
            self._model_suite = ModelSuite.load(self.models_dir)
        return self._model_suite

    @property
    def segmenter(self) -> Segmenter:
        if self._segmenter is None:
            self._segmenter = Segmenter()
        return self._segmenter

    def load_chart(self, ssc_path: str, difficulty: Optional[int] = None, play_style: str = 'singles') -> ChartStruct:
        song = SongSSC.from_file(ssc_path)
        if difficulty is not None:
            stepchart = song.get_stepchart(difficulty=difficulty, play_style=play_style)
        else:
            stepchart = song.stepcharts[0]
        return ChartStruct.from_stepchart_ssc(stepchart)

    def load_chartstruct(self, csv_path: str) -> ChartStruct:
        return ChartStruct.from_file(csv_path)

    def analyze_chart(self, cs: ChartStruct, skills: list[str] = None) -> ChartAnalysisResult:
        if skills is None:
            skills = ['drill', 'run', 'bracket', 'twist_90', 'jack', 'footswitch']

        annotate_skills(cs, skill_names=skills)

        sections = self.segmenter.segmentation(cs)

        cs, fcs, pred_limbs = predict_limbs(cs, self.model_suite)
        cs.add_limb_annotations(fcs.pred_coords, pred_limbs, 'Predicted limbs')

        return ChartAnalysisResult(
            chartstruct=cs,
            sections=sections,
            pred_limbs=pred_limbs,
        )

    def predict_difficulty(self, section) -> float:
        if self._difficulty_predictor is None:
            self._difficulty_predictor = DifficultySegmentModelPredictor.load(self.models_dir)
        return self._difficulty_predictor.predict(section)

    def to_visualization_json(self, cs: ChartStruct, output_path: str):
        cjs = ChartJsStruct.from_chartstruct(cs)
        cjs.to_json(output_path)


def find_ssc_files(root_dir: str, pattern: str = '*.ssc') -> list[str]:
    from glob import glob
    return glob(os.path.join(root_dir, '**', pattern), recursive=True)


def find_chartstructs(root_dir: str, level: Optional[int] = None) -> list[str]:
    files = []
    for fn in os.listdir(root_dir):
        if fn.endswith('.csv'):
            if level is not None:
                if fn.startswith(f'S{level}') or fn.startswith(f'D{level}'):
                    files.append(os.path.join(root_dir, fn))
            else:
                files.append(os.path.join(root_dir, fn))
    return files


def export_chart_summary(cs: ChartStruct) -> dict:
    return {
        'source_file': cs.source_file,
        'level': cs.get_chart_level(),
        'sord': cs.singles_or_doubles(),
        'num_lines': len(cs.df),
        'duration': cs.df['Time'].max() - cs.df['Time'].min(),
        'metadata': cs.metadata,
    }


__all__ = [
    'PiuAnnotateAgent',
    'ChartAnalysisResult',
    'find_ssc_files',
    'find_chartstructs',
    'export_chart_summary',
]