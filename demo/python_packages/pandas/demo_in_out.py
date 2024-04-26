"""Getting data in/out
======================

"""

import pandas as pd
import numpy as np


# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPandas.png'
timestamps = pd.Series(
    np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000)
)
timestamps = timestamps.cumsum()
dataFrame = pd.DataFrame(
    np.random.randn(1000, 4), index=timestamps.index, columns=["A", "B", "C", "D"]
)
dataFrame = dataFrame.cumsum()

dataFrame.to_csv("foo.csv")
pd.read_csv("foo.csv")

###############################################################################
dataFrame.to_excel("foo.xlsx", sheet_name="Sheet1")
pd.read_excel("foo.xlsx", "Sheet1", index_col=None, na_values=["NA"])
