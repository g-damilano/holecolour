import warnings

from .config.schema import PipelineConfig

warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '2\.10\.2' or newer of 'numexpr'.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '1\.4\.2' or newer of 'bottleneck'.*",
    category=UserWarning,
)

__all__ = ["PipelineConfig"]
__version__ = "0.48.0"
