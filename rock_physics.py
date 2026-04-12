from __future__ import annotations

import numpy as np
import pandas as pd


def compute_elastic_properties(
    df: pd.DataFrame,
    vp_col: str,
    vs_col: str,
    rho_col: str,
    youngs_col: str = "Youngs_modulus",
    poisson_col: str = "Poissons_ratio",
    bulk_col: str = "Bulk_modulus",
    shear_col: str = "Shear_modulus",
) -> pd.DataFrame:
    result = df.copy()

    vp = pd.to_numeric(result[vp_col], errors="coerce")
    vs = pd.to_numeric(result[vs_col], errors="coerce")
    rho = pd.to_numeric(result[rho_col], errors="coerce")

    vp2 = vp**2
    vs2 = vs**2

    shear_modulus = rho * vs2
    bulk_modulus = rho * (vp2 - (4.0 / 3.0) * vs2)
    poisson_ratio = (vp2 - 2 * vs2) / (2 * (vp2 - vs2))
    youngs_modulus = 2 * shear_modulus * (1 + poisson_ratio)

    bulk_modulus = bulk_modulus.where(bulk_modulus > 0)

    result[shear_col] = shear_modulus
    result[bulk_col] = bulk_modulus
    result[poisson_col] = poisson_ratio
    result[youngs_col] = youngs_modulus

    invalid_mask = (vp <= 0) | (vs <= 0) | (rho <= 0) | (vp <= vs)
    result.loc[invalid_mask, [shear_col, bulk_col, poisson_col, youngs_col]] = np.nan

    return result
