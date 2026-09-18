#!/usr/bin/env python
"""Predict MLGF self-energy and QP energies from a DFT checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from mlgf.lib.ml_helper import sigma_lo_mo
from mlgf.workflow.get_ml_info import get_properties

PUBLISHED_AC_IDX = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23]


def predict_qpe(checkpoint: Path, model: Path) -> dict[str, np.ndarray]:
    estimator = joblib.load(model)
    sigma_saiao = estimator.predict_full_sigma(str(checkpoint))
    mlf = estimator.pdset[0]
    omega = np.asarray(mlf['omega_fit'])
    if sigma_saiao.shape[-1] != len(omega):
        raise ValueError(
            f'self-energy has {sigma_saiao.shape[-1]} frequencies but checkpoint provides {len(omega)}; '
            'refuse QP continuation with an unmatched grid'
        )
    sigma_mo = sigma_lo_mo(sigma_saiao, mlf['C_saiao_mo'])
    result = get_properties(
        sigma_mo, mlf, np.linspace(-1.0, 1.0, 201), 0.01,
        properties='q', ac_idx=PUBLISHED_AC_IDX,
    )
    return {
        'qpe_hartree': np.asarray(result['qpe']),
        'sigma_saiao': np.asarray(sigma_saiao),
        'sigma_mo': np.asarray(sigma_mo),
        'mo_energy_hartree': np.asarray(mlf['mo_energy']),
        'mo_occ': np.asarray(mlf['mo_occ']),
        'omega_fit_hartree': omega,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = predict_qpe(args.checkpoint, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    finite = int(np.isfinite(output['qpe_hartree']).sum())
    print(f'Wrote {args.output}; finite QP roots: {finite}/{len(output["qpe_hartree"])}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
