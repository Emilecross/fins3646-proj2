from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# from project2 import config as cfg
# from project2 import util

import config as cfg
import util

def rename_cols(df: pd.DataFrame, prc_col: str = "adj_close"):
    """ Rename the columns of df in place.

    Normalise the names of columns in a dataframe such that they are in
    snake case with no leading or trailing white spaces. The prc_col
    parameter indicates which column should be renamed to 'price'.
    This function should be used in read_dat and read_csv.

    Parameters
    ----------
    df: frame
        A data frame with daily prices (open, close, etc...) for different
        tickers.


    prc_col: str
        Which price to use (close, open, etc...).
    """
    df.columns = [
        str(x).lower().strip().replace("-", "_").replace(" ", "_") for x in df.columns
    ]
    df.rename(columns={prc_col: "price"}, inplace=True)

def read_dat(pth, prc_col: str = "adj_close"):
    """ Returns a data frame with the relevant information from the dat file
    Parameters
    ----------
    pth: str
        Location of the .dat file to be read
    prc_col: str
        Which price column to use (close, open, etc...)
    Returns
    -------
    frame: 
    A dataframe with columns:
        #   Column   
    ---  ------   
        0   date     
        1   ticker   
        2   price    
    """
    rtn_data = []
    TEMPLATE = {"Ticker": "TMP",
                'Volume': 14,
                'Open': 6,
                'Close': 10,
                'High': 20,
                'Low': 20,
                'Adj Close': 19,
                'Date': "01-01-2001"}
    with open(pth) as tic_data:
        first = True
        for data_point in tic_data:
            if first:
                first = False
                continue

            row = None
            if len(data_point.split(",")) == 8:
                row = data_point.split(",")
            elif len(data_point.split("\t")) == 8:
                row = data_point.split("\t")
            elif len(data_point.split(" ")) == 8:
                row = data_point.split(" ")

            if row:
                insert = TEMPLATE.copy()
                for i, k in enumerate(insert.keys()):
                    insert[k] = row[i].strip('\'\" ')

                insert["Ticker"] = str(insert["Ticker"])
                insert["Volume"] = float(insert["Volume"])
                insert["Adj Close"] = float(insert["Adj Close"])
                insert["Close"] = float(insert["Close"])
                insert["Open"] = float(insert["Open"])
                insert["High"] = float(insert["High"])
                insert["Date"] = pd.to_datetime(insert["Date"])
                rtn_data.append(insert)

    df = pd.DataFrame(rtn_data)
    rename_cols(df, prc_col)
    return_df = df[["ticker", "date", "price"]]
    return return_df

def read_csv(pth, ticker: str, prc_col: str = "adj_close"):
    """Returns a DF with the relevant information from the CSV file `pth`

    Parameters
    ----------
    pth: str
        Location of the CSV file to be read
    ticker:
        Relevant ticker
    prc_col: str
        Which price column to use (close, open, etc...)
    Returns
    -------
    frame:
        A dataframe with columns:
            #   Column
        ---  ------
            0   date
            1   ticker
            2   price
    """
    df = pd.read_csv(pth)
    rename_cols(df, prc_col)
    df["ticker"] = ticker
    return_df = df[["ticker", "date", "price"]]
    df["date"] = pd.to_datetime(df["date"])
    return return_df


def read_files(
    csv_tickers: list[str] | None = None,
    dat_files: list[str] | None = None,
    prc_col: str = "adj_close",
):
    """Read CSV and DAT files. If an observation [ticker, price] is
    present in both files, prioritize CSV

    Parameters
    ----------
    csv_ticker: list, str, optional

    dat_files: list, str, optional

    prc_col: str
        Which price to use (close, open, etc...).

    Returns
    -------
    frame:
        A dataframe with columns:

         #   Column
        ---  ------
         0   date
         1   ticker
         2   price
    """
    # Initialise an empty DataFrame
    df_list = []

    if csv_tickers:
        for ticker in csv_tickers:
            if ticker.endswith("_prc.csv"):
                ticker = ticker.split("_")[0]
            ticker = ticker.lower()
            pth = os.path.join(cfg.DATADIR ,f"{ticker}_prc.csv")
            df_list.append(read_csv(pth, ticker, prc_col))

    if dat_files:
        for dat_name in dat_files:
            if dat_name.endswith(".dat"):
                dat_name = dat_name.split(".")[0]
            dat_name = dat_name.lower()
            pth = os.path.join(cfg.DATADIR ,f"{dat_name}.dat")
            df_list.append(read_dat(pth, prc_col))

    return pd.concat(df_list).drop_duplicates()


def calc_monthly_ret_and_vol(df: pd.DataFrame):
    """Compute monthly returns and volatility for each ticker in `df`.
    Parameters
    ----------
    df: frame
        A data frame with columns
         #   Column
        ---  ------
         0   date
         1   ticker
         2   price
    Returns
    -------
    frame:
        A data frame with columns
         #   Column
        ---  ------
         0   mdate
         1   ticker
         2   mret
         3   mvol
        where
            mdate is a string with format YYYY-MM

            ticker is the ticker (uppercase, no spaces, no quotes)

            mret is the monthly return

            mvol is the monthly volatility. Computed as the
                standard deviation of daily returns * sqrt(21)
    Notes
    -----
    Assume no gaps in the daily time series of each ticker
    """
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    monthly_stats = []

    for ticker, ticker_df in df.groupby("ticker"):
        monthly_prices = ticker_df['price'].resample('ME').last()
        monthly_returns = monthly_prices.pct_change().dropna()

        daily_returns = ticker_df['price'].pct_change().dropna()
        monthly_volatility = daily_returns.resample('ME').std() * np.sqrt(21)

        for date in monthly_returns.index:
            monthly_stats.append({
                "mdate": date.strftime("%Y-%m"),
                "ticker": ticker.upper(),
                "mret": monthly_returns[date],
                "mvol": monthly_volatility.loc[date]
            })

    return pd.DataFrame(monthly_stats)


def main(
    csv_tickers: list | None = None,
    dat_files: list | None = None,
    prc_col: str = "adj_close",
):
    """Perform the main analysis. Regressing month returns on lagged monthly
    volatility.

    Parameters
    ----------
    csv_tickers: list
        A list of strings, where each string is the ticker of a stock for which
        the data is in a CSV file.

    dat_files: list
        A list of strings, where each string is the name of a dat file.

    prc_col: str
        The name of the column in which price data is to be read.

    Returns
    -------
    None

    Notes
    -----
    The function should print the summary results of a linear regression provided by
    the statsmodels package.
    """
    pass

def test_step_1_2():
    result = pd.read_csv(os.path.join(cfg.DATADIR, 'res.csv')).equals(pd.read_csv(os.path.join(cfg.DATADIR, 'sample.csv')))
    print(f'Dataframes are the same: {result}')

if __name__ == "__main__":
    calc_monthly_ret_and_vol(read_files(csv_tickers=["tsla"], dat_files=["data1"])).to_csv(os.path.join(cfg.DATADIR, 'res.csv'), index=False)
    # test_step_1_2()