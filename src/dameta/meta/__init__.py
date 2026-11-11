"""Meta-learned optimizers: Meta-B delta chain (teacher-relative stretch) and Meta-B supervised (milestone 4)."""

from .meta_b import MetaB, MetaBConfig, control_stats, run_meta_b, train_meta_b
from .meta_b_sup import MetaBSup, SupConfig, build_sup, collect_teacher_pairs, run_meta_b_sup
from .networks import GradNet, IterationEmbedding, StretchNet

__all__ = ["MetaB", "MetaBConfig", "control_stats", "run_meta_b", "train_meta_b", "MetaBSup", "SupConfig",
           "build_sup", "collect_teacher_pairs", "run_meta_b_sup", "GradNet", "IterationEmbedding", "StretchNet"]
