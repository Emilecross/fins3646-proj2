""" config.py

Configuration options for the project2 package         


"""
# IMPORTANT: Please do NOT modify this file


import os
import toolkit_config as tcfg

ROOTDIR = os.path.join(tcfg.PRJDIR, 'project2')
DATADIR = os.path.join(ROOTDIR, 'data')

TICMAP = {
        'AAL'    : 'American Airlines Group Inc',
        'AAPL'   : 'Apple Inc.',
        'ABBV'   : 'AbbVie Inc.',
        'BABA'   : 'Alibaba Group Holding Limited',
        'BAC'    : 'Bank of America Corporation',
        'CSCO'   : 'Cisco Systems, Inc.',
        'DAL'    : 'Delta Air Lines, Inc.',
        'DIS'    : 'The Walt Disney Company',
        'FB'     : 'Facebook, Inc.',
        'GE'     : 'General Electric Company',
        'INTC'   : 'Intel Corporation',
        'JNJ'    : 'Johnson & Johnson',
        'KO'     : 'The Coca-Cola Company',
        'MSFT'   : 'Microsoft Corporation',
        'NVDA'   : 'NVIDIA Corporation',
        'ORCL'   : 'Oracle Corporation',
        'PFE'    : 'Pfizer Inc.',
        'PG'     : 'The Procter & Gamble Company',
        'PYPL'   : 'PayPal Holdings, Inc.',
        'T'      : 'AT&T Inc. (T)',
        'TSLA'   : 'Tesla, Inc.',
        'TSM'    : 'Taiwan Semiconductor Manufacturing Company Limited',
        'V'      : 'Visa Inc.',
        }

TICKERS = sorted(TICMAP.keys())
    
