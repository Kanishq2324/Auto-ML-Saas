"""Shared configuration for the auto ML engine"""

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_CV_FOLDS = 5

SUPPORTED_TASKS = {
    "auto",
    "classification",
    "regression"
}

SUPPORTED_TUNING_MODES = {
    "fast",
    "balanced",
    "thorough"
}


"""Shared settings remain consistent throughout all modules."""