"""Grouping data
================

"""
import pandas as pd
import numpy as np

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPandas.png'

###############################################################################
# **Concat**
# 
# Create a dataFrame
dataFrame = pd.DataFrame(np.random.randn(10, 4))

###############################################################################
# break in pieces
pieces = [dataFrame[:3], dataFrame[3:7], dataFrame[7:]]

###############################################################################
pd.concat(pieces)

###############################################################################
# **Join**
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})

###############################################################################
pd.merge(left, right, on="key")

###############################################################################
# **Grouping**
dataFrame = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
        "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)
dataFrame.groupby("A").sum()

###############################################################################
dataFrame.groupby(["A", "B"]).sum()