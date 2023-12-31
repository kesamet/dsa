"""
Utility functions for problem 1
"""
import logging

import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

logging.basicConfig(level=logging.INFO)
sns.set()

FILENAME = "./DAE002/DS1-assessment-RMD UST Yield Data.xlsx"


def load_yields() -> pd.DataFrame:
    """Loads yield data.

    Returns:
        pd.DataFrame: A DataFrame containing the yields.
    """
    sheet_name = "DGS10"
    df = pd.read_excel(
        FILENAME,
        sheet_name=sheet_name,
        index_col="observation_date",
        parse_dates=True,
        skiprows=10,
    )
    logging.info(f"Loading {sheet_name}")

    sheet_names = [
        "Fed Funds Effective Rate",
        "DGS1MO",
        "DGS3MO",
        "DGS6MO",
        "DGS1",
        "DGS2",
        "DGS3",
        "DGS5",
        "DGS7",
        "DGS20",
        "DGS30",
    ]
    for sheet_name in sheet_names:
        _df = pd.read_excel(
            FILENAME,
            sheet_name=sheet_name,
            index_col="observation_date",
            parse_dates=True,
            skiprows=10,
        )
        df = df.join(_df, how="inner")
        logging.info(f"Loading {sheet_name}")
    logging.info(f"Final df shape: {df.shape}")
    return df


def load_econs() -> pd.DataFrame:
    """Loads economic data.

    Returns:
        pd.DataFrame: A dataframe containing economic data.
    """
    sheet_name = "Initial Claims (ICSA)"  # weekly
    df = pd.read_excel(
        FILENAME,
        sheet_name=sheet_name,
        index_col="observation_date",
        parse_dates=True,
        skiprows=10,
    )
    df = df[["ICSA"]].resample("M").mean()
    df.index += pd.Timedelta(
        days=1
    )  # add 1 day so that observation date will be on the 1st of each month
    logging.info(f"Loading {sheet_name}")

    sheet_names = [
        "Capacity Utilization (Fred)",
        "Industrial Production",
        "CPI (Level, SA, Fred)",
        "PCE (Level, SA, Fred, SAAR)",
        "IPI (All Commods, Index, NSA)",
        "Total Nonfarm (Fred)",
        "Unemployment Rate (SA, Fred)",
        "Adv Retail Sales (SA, Mil)",
    ]

    for sheet_name in sheet_names:
        _df = pd.read_excel(
            FILENAME,
            sheet_name=sheet_name,
            index_col="observation_date",
            parse_dates=True,
            skiprows=10,
        )
        df = df.join(_df, how="inner")
        logging.info(f"Loading {sheet_name}")
    logging.info(f"Final df shape: {df.shape}")
    return df


def evaluate(data: pd.DataFrame) -> None:
    """
    Evaluates the model performance on the given data.

    Args:
        data (pd.DataFrame): The data to evaluate the model on.
    """
    y_true = data["y"].values
    y_pred = data["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MAE = {mae:.4f}")
    print(f"MAPE = {mape:.4f}")
    print(f"R2 = {r2:.4f}")

    _ = sns.jointplot(x="y", y="yhat", data=data, kind="reg", color="0.4")
