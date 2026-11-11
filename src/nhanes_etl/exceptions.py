class PipelineError(RuntimeError):
    """Base pipeline error."""


class ConfigError(PipelineError):
    """Invalid cycle dictionary or settings."""


class DownloadError(PipelineError):
    """Download failed or integrity check failed."""


class SchemaAuditError(PipelineError):
    """Raw file schema failed a locked audit."""


class MergeError(PipelineError):
    """Unsafe or invalid merge."""
