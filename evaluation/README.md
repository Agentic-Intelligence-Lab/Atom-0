# Offline evaluation tools

- [VLA trajectory evaluation](vla/README_CN.md): multi-checkpoint Piper tooling.
- [Robot-only VLA baseline evaluation](baseline_vla/README_CN.md): retained
  evaluator for the historical real-only checkpoint and its Unified80 mapping.

These are offline tools, not new real-robot results or a certified evaluator for
every paper checkpoint. Select the matching dataset, model configuration,
normalization, action mapping and split before running them. See the
[paper evaluation protocol](../docs/evaluation.md).

Run the suites in separate Python processes: both retained tools use an
`evaluate_validation` module name.

```bash
PYTHONPATH=src:packages/openpi-client/src python -m pytest evaluation/vla/tests -q -o addopts=''
PYTHONPATH=src:packages/openpi-client/src python -m pytest evaluation/baseline_vla/tests -q -o addopts=''
```
