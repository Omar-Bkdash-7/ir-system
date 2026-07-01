from common.exceptions import (
    IRSystemException,
    PreprocessingError,
    ModelLoadError,
)


class ResourceNotFoundError(PreprocessingError):
    pass


class TokenizationError(PreprocessingError):
    pass
