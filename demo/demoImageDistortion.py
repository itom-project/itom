from cameraToolbox import distortion_test
from itom import *

"""in the folder itom_packages, there is a package 'cameraToolbox'.
This mainly contains methods to simulate distorted images
and calculate the distortion of acquired grids.

To simplify the access to this script, this demo is added.
For more information, see the module distortion_test.py"""


def demo_distortion():
    distortion_test.distortionTest()


if __name__ == "__main__":
    demo_distortion()
