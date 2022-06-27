"""Operations
=============

"""
import pandas as pd
import numpy as np

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPandas.png'

dates = pd.date_range("20220501", periods=6)
dataFrame = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

###############################################################################
# **Statistics**
dataFrame.mean()

###############################################################################
# Mean value of axis ``1``:
dataFrame.mean(1)

###############################################################################
series = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)

###############################################################################
dataFrame.sub(series, axis="index")

###############################################################################
# **Apply**
dataFrame.apply(np.cumsum)

###############################################################################
dataFrame.apply(lambda x: x.max() - x.min())

###############################################################################
# **Histogramming**
series = pd.Series(np.random.randint(0, 7, size=10))
series.value_counts()

###############################################################################
# **String methods**
series = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
series.str.lower()