"""Co-training extensions for multi-dataset RLDS training with train/val splits.

This package is a self-contained fork of the relevant openpi RLDS data-loading /
training pieces. It does NOT modify any existing openpi files; it only imports the
stable, unchanged helpers (transforms, model, sharding, RLDSDataLoader, DataLoaderImpl,
etc.) and re-implements the parts that need new behavior:

  * multi-dataset weighted mixture with per-dataset train/val splits
  * validation flow-matching loss (fixed-seed and multi-sample-average modes)
  * validation action MSE (a la EgoVerse offline metric), per-dataset + aggregate
"""
