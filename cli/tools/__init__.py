from __future__ import annotations
"""
Agent Utilities for piu-annotate

Este modulo proporciona funciones de alto nivel para que agentes IA
interactuen con el sistema de anotacion de charts de PIU.

Funciones principales:
- load_and_analyze(): Carga y analiza un chart completo
- batch_process(): Procesa multiples charts en paralelo
- get_chart_stats(): Obtiene estadisticas de un chart
- compare_charts(): Compara dos charts
"""

from typing import Optional, Literal
from dataclasses import dataclass
import os
from glob import glob

from piu_annotate.formats.sscfile import SongSSC
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.segment.segment import Segmenter
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict as predict_limbs
from piu_annotate.formats.jsplot import ChartJsStruct


@dataclass
class ChartInfo:
    source_file: str
    level: int
    sord: str
    num_lines: int
    duration: float
    num_sections: int
    skills_present: dict

    def to_dict(self):
        return {
            'source_file': self.source_file,
            'level': self.level,
            'sord': self.sord,
            'num_lines': self.num_lines,
            'duration': self.duration,
            'num_sections': self.num_sections,
            'skills_present': self.skills_present,
        }


def load_ssc(ssc_path: str) -> SongSSC:
    """Carga un archivo SSC."""
    return SongSSC.from_file(ssc_path)


def get_stepchart(song: SongSSC, difficulty: Optional[int] = None, play_style: str = 'singles') -> StepchartSSC:
    """Obtiene un stepchart especifico de un song."""
    if difficulty is not None:
        return song.get_stepchart(difficulty=difficulty, play_style=play_style)
    return song.stepcharts[0]


def to_chartstruct(stepchart) -> ChartStruct:
    """Convierte un stepchart SSC a ChartStruct."""
    return ChartStruct.from_stepchart_ssc(stepchart)


def annotate_chart_skills(cs: ChartStruct, skills: list[str]) -> None:
    """Anota skills en el chart."""
    annotate_skills(cs, skill_names=skills)


def segment_chart(cs: ChartStruct) -> list:
    """Segmenta el chart en secciones."""
    segmenter = Segmenter()
    return segmenter.segmentation(cs)


def predict_limbs_on_chart(cs: ChartStruct, model_suite: ModelSuite) -> tuple:
    """Predice extremidades en el chart. Retorna (cs, fcs, pred_limbs)."""
    return predict_limbs(cs, model_suite)


def add_limb_annotations(cs: ChartStruct, fcs, pred_limbs: list, col_name: str = 'Predicted limbs') -> None:
    """Anade las anotaciones de extremidades al ChartStruct."""
    cs.add_limb_annotations(fcs.pred_coords, pred_limbs, col_name)


def to_json(cs: ChartStruct, output_path: str) -> None:
    """Exporta el chart a JSON para visualizacion."""
    cjs = ChartJsStruct.from_chartstruct(cs)
    cjs.to_json(output_path)


def get_chart_info(cs: ChartStruct) -> ChartInfo:
    """Obtiene informacion del chart."""
    duration = cs.df['Time'].max() - cs.df['Time'].min()

    skills_present = {}
    for col in cs.df.columns:
        if col.startswith('__'):
            continue
        if cs.df[col].dtype == bool:
            if cs.df[col].any():
                skills_present[col] = int(cs.df[col].sum())

    return ChartInfo(
        source_file=cs.source_file,
        level=cs.get_chart_level(),
        sord=cs.singles_or_doubles(),
        num_lines=len(cs.df),
        duration=duration,
        num_sections=0,
        skills_present=skills_present,
    )


def load_and_analyze(
    ssc_path: str,
    models_dir: str,
    difficulty: Optional[int] = None,
    play_style: str = 'singles',
    skills: Optional[list[str]] = None,
) -> ChartInfo:
    """Funcion principal: carga y analiza un chart completo.

    Args:
        ssc_path: Ruta al archivo .ssc
        models_dir: Ruta al directorio de modelos
        difficulty: Nivel de dificultad (None = primer stepchart)
        play_style: 'singles' o 'doubles'
        skills: Lista de skills a anotar

    Returns:
        ChartInfo con informacion del chart
    """
    if skills is None:
        skills = ['drill', 'run', 'bracket', 'twist_90', 'jack', 'footswitch', 'anchor_run']

    song = load_ssc(ssc_path)
    stepchart = get_stepchart(song, difficulty=difficulty, play_style=play_style)
    cs = to_chartstruct(stepchart)

    annotate_chart_skills(cs, skills)
    sections = segment_chart(cs)

    model_suite = ModelSuite.load(models_dir)
    cs, fcs, pred_limbs = predict_limbs_on_chart(cs, model_suite)
    add_limb_annotations(cs, fcs, pred_limbs)

    info = get_chart_info(cs)
    info.num_sections = len(sections)
    return info


def find_ssc_files(directory: str, recursive: bool = True) -> list[str]:
    """Encuentra todos los archivos .ssc en un directorio."""
    pattern = '**/*.ssc' if recursive else '*.ssc'
    return glob(os.path.join(directory, pattern), recursive=recursive)


def batch_analyze(
    ssc_files: list[str],
    models_dir: str,
    skills: Optional[list[str]] = None,
    workers: int = 4,
) -> list[ChartInfo]:
    """Procesa multiples charts en paralelo."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if skills is None:
        skills = ['drill', 'run', 'bracket', 'twist_90', 'jack', 'footswitch']

    def process_one(ssc_path):
        try:
            return load_and_analyze(ssc_path, models_dir, skills=skills)
        except Exception as e:
            return None

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one, f) for f in ssc_files]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    return results


__all__ = [
    'load_ssc',
    'get_stepchart',
    'to_chartstruct',
    'annotate_chart_skills',
    'segment_chart',
    'predict_limbs_on_chart',
    'add_limb_annotations',
    'to_json',
    'get_chart_info',
    'load_and_analyze',
    'find_ssc_files',
    'batch_analyze',
    'ChartInfo',
]