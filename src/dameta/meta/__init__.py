"""Meta-learned optimizers (M4: Meta-B delta + supervised; M5: learned-optimization unroll and the meta grid)."""

from .meta_a import LearnedOptimization, UnrollConfig, build_unroll, obs_features, run_meta_a
from .meta_b import MetaB, MetaBConfig, control_stats, run_meta_b, train_meta_b
from .meta_b_sup import MetaBSup, SupConfig, build_sup, collect_teacher_pairs, run_meta_b_sup
from .networks import FreeUpdateNet, GradNet, IterationEmbedding, StretchNet

__all__ = ["LearnedOptimization", "UnrollConfig", "build_unroll", "obs_features", "run_meta_a", "MetaB",
           "MetaBConfig", "control_stats", "run_meta_b", "train_meta_b", "MetaBSup", "SupConfig", "build_sup",
           "collect_teacher_pairs", "run_meta_b_sup", "FreeUpdateNet", "GradNet", "IterationEmbedding", "StretchNet"]
