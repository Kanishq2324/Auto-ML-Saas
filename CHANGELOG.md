# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-07-29

### Added

- Reusable Python package under `src/automl_engine`
- CSV dataset loading and validation
- Dataset inspection and safe cleaning
- Automatic task detection
- Classification and regression target preparation
- Numerical and categorical feature detection
- Numerical and categorical preprocessing pipelines
- Classification and regression model registries
- Cross-validation and baseline leaderboards
- Randomized hyperparameter tuning
- Final model selection with baseline fallback
- Final untouched test-set evaluation
- Classification reports and confusion matrices
- Regression residual output
- Model and report artifact persistence
- Saved-model loading
- DataFrame and CSV prediction support
- Classification label decoding and probabilities
- Unit tests
- End-to-end classification and regression integration tests
- Thin demonstration notebook

### Current limitations

- CSV tabular data only
- Classification and regression only
- No web API or graphical interface
- No time-series, clustering, or text/image modeling