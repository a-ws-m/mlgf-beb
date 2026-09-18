# BEB-compatible MLGF execution profile

This fork uses the public `lijiachen417/fcdmft-public` interface at commit
`0e6a16da8508187a96d3a43bb91368e0c4e6c028`. Use Python 3.11 and
`requirements-beb-lock.txt`; clone that fcdmft source and libdmet_preview at
recorded commits, put both source roots plus this checkout on `PYTHONPATH`,
then install this checkout editable.

The public upstream MLGF release has no dependency lock and imports fcdmft
modules absent from its documented fcdmft repository. This fork repairs the
DFT-only path and records the 30-point imaginary-frequency grid used by
`GWAC.nw2=30`; it refuses QP continuation when model and grid lengths differ.

```bash
python mlgf/workflow/generate.py --calc dft --basis ccpvdz --xyz_file molecule.xyz --chk_file molecule.chk --json_spec pbe0.json
python mlgf/workflow/predict_qpe.py --checkpoint molecule.chk --model examples/pretrained_models/qm9_model.joblib --output prediction.npz
```
