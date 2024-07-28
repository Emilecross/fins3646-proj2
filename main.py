""" scaffold for project2
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


from project2 import config as cfg
from project2 import util


def rename_cols(
        df: pd.DataFrame,
        prc_col: str = 'adj_close',
        ) -> None:
    """ Rename the columns of `df` in place.

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
    pass

def read_dat(
        pth,
        prc_col: str = 'adj_close',
        ) -> pd.DataFrame:
    """ Returns a data frame with the relevant information from the .dat file
    `pah`


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
    pass



def read_csv(
        pth,
        ticker: str,
        prc_col: str = 'adj_close',
        ) -> pd.DataFrame:
    """ Returns a DF with the relevant information from the CSV file `pth`

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
    pass
    

def read_files(
        csv_tickers: list | None = None,
        dat_files: list | None = None,
        prc_col: str = 'adj_close',
        ):
    """ Read CSV and DAT files. If an observation [ticker, price] is
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
    pass


def calc_monthly_ret_and_vol(df):
    """ Compute monthly returns and volatility for each ticker in `df`.

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
    pass


def main(
        csv_tickers: list | None = None,
        dat_files: list | None = None,
        prc_col: str = 'adj_close',
        ):
    """ Perform the main analysis. Regressing month returns on lagged monthly
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


if __name__ == "__main__":
    pass
    



