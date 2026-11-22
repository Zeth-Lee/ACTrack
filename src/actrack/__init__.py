__version__ = "0.1.0"
__all__ = ["PipelineConfig", "ACTrackPipeline", "BBox", "load_config"]


def __getattr__(name):
    if name in {"PipelineConfig", "load_config"}:
        from . import config
        return getattr(config, name)
    if name == "ACTrackPipeline":
        from .pipeline import ACTrackPipeline
        return ACTrackPipeline
    if name == "BBox":
        from .types import BBox
        return BBox
    raise AttributeError(name)
