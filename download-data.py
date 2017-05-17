#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:58:26 2017

@author: gabegm
"""

import quandl

# quandl.get(":database_code/:dataset_code", returns = ":return_format")
quandl.ApiConfig.api_key = 'GY56ffY8tRJKuZyfYVsH'

# When returns is omitted, a pandas dataframe is returned
data = quandl.get("WIKI/FB")

# quandl.Dataset(":database_code/:dataset_code").data_fields()
metadata = quandl.Dataset("WIKI/FB").data_fields()

# Multiple datasets in one call, comma-delimit their codes and put them in an array
merged_data = quandl.get(["WIKI/FB", "EOD/AAPL", "WIKI/MSFT"])
