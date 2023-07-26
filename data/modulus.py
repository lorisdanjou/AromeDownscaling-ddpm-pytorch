import pandas as pd
import numpy as np


def modulus_arrays(a, b):
    """
    Computes the modulus of two numpy arrays
    """
    return np.sqrt(a**2 + b**2)


def compute_modulus(wind_df):
    """
    Computes the modulus of the wind for a DataFrame

    Args:
        wind_df (Dataframe): contains the components of the wind speed

    Returns:
        DataFrame: contains the modulus of the wind speed
    """
    modulus_df = pd.DataFrame(
        [],
        columns = ["dates", "echeances", "modulus"]
    )

    for i in range(len(wind_df)):
        modulus_df.loc[len(modulus_df)] = [
            wind_df.dates.iloc[i], 
            wind_df.echeances.iloc[i], 
            modulus_arrays(wind_df.u10.iloc[i], wind_df.v10.iloc[i])
        ]

    return modulus_df