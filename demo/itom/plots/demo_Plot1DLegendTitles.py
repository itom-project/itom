"""1D Legend title
======================

This demo shows how to define ``dataObject`` tags, which are used
as ``legendTitles`` in the 1D plot. You have to set the tags ``legendTitle0``
``legendTitle1``, ``legendTitle2``, ..., according to the curve index and
the legend label text."""

from itom import dataObject
from itom import plot1
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPlot1DLegendTitle.png'

dObj = dataObject.rand([2, 100])
dObj.setTag("legendTitle0", "title of the first curve")
dObj.setTag("legendTitle1", "title of the second curve")

print(dObj.tags)
plot1(dObj, properties={"legendPosition": "Right"})

###############################################################################
# .. image:: ../../_static/demoPlot1DLegendTitle_1.png
#    :width: 100%
