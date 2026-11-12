class IRSystemException(Exception):
    pass

class IndexNotFoundError(IRSystemException):
    pass

class InvalidQueryError(IRSystemException):
    pass

class RetrievalError(IRSystemException):
    pass

class RankingError(IRSystemException):
    pass

class PreprocessingError(IRSystemException):
    pass

class DatasetValidationError(IRSystemException):
    pass

class ModelLoadError(IRSystemException):
    pass

class ServiceError(IRSystemException):
    pass
