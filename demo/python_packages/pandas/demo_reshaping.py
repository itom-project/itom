"""Reshaping data
=================

"""
import pandas as pd
import numpy as np

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPandas.png'

tuples = list(
    zip(
        *[
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
    )
)
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
dataFrame = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
dataFrame2 = dataFrame[:4]

###############################################################################
# **Stack**
stacked = dataFrame2.stack()

###############################################################################
stacked.unstack()

###############################################################################
stacked.unstack(1)

###############################################################################
stacked.unstack(0)

###############################################################################
# **Pivot tables**
dataFrame = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 3,
        "B": ["A", "B", "C"] * 4,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
        "D": np.random.randn(12),
        "E": np.random.randn(12),
    }
)

###############################################################################
pd.pivot_table(dataFrame, values="D", index=["A", "B"], columns=["C"])

###############################################################################
# **Time series**
indexData = pd.date_range("1/5/2022", periods=100, freq="S")
timeStemps = pd.Series(np.random.randint(0, 500, len(indexData)), index=indexData)
timeStemps.resample("5Min").sum()

###############################################################################
timeStempsUTC = timeStemps.tz_localize("UTC")

###############################################################################
timeStempsUTC.tz_convert("US/Eastern")

###############################################################################
ps = timeStemps.to_period()

###############################################################################
ps.to_timestamp()

###############################################################################
prng = pd.period_range("1990Q1", "2000Q4", freq="Q-NOV")
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq("M", "e") + 1).asfreq("H", "s") + 9