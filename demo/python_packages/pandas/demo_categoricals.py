"""Categoricals
===============

"""
import pandas as pd

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPandas.png'

dataFrame = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]})

###############################################################################
dataFrame["grade"] = dataFrame["raw_grade"].astype("category")

###############################################################################
dataFrame["grade"].cat.rename_categories(["very good", "good", "very bad"])
dataFrame["grade"] = dataFrame["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
dataFrame["grade"]

###############################################################################
dataFrame.sort_values(by="grade")

###############################################################################
dataFrame.groupby("grade").size()

###############################################################################

###############################################################################

