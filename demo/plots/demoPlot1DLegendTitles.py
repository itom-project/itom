"""This demo shows how to define dataObject tags, which are used 
as legendTitles in the 1D plot. """
from itom import *


def demo_plotLegends():
    d = dataObject.rand([2, 100])
    d.setTag("legendTitle0", "title of the first curve")
    d.setTag("legendTitle1", "title of the second curve")
    plot(d, "1d", properties={"legendPosition": "Right"})


if __name__ == "__main__":
    demo_plotLegends()
