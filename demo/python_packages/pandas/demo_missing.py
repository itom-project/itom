"""Missing data
===============

"""
import pandas as pd
import numpy as np

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPandas.png'

dates = pd.date_range("20220501", periods=6)
dataFrame = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

###############################################################################
# **Missing data**
dataFrame1 = dataFrame.reindex(index=dates[0:4], columns=list(dataFrame.columns) + ["E"])
dataFrame1.loc[dates[0] : dates[1], "E"] = 1

###############################################################################
dataFrame1.dropna(how="any")

###############################################################################
dataFrame1.fillna(value=5)

###############################################################################
pd.isna(dataFrame1)
