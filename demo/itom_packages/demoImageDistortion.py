"""cameraToolbox
===========

In the folder ``itom_packages``, there is a package ``cameraToolbox``.
This mainly contains methods to simulate distorted images
and calculate the distortion of acquired grids.

To simplify the access to this script, this demo is added.
For more information, see the module ``distortion_test.py``."""

from cameraToolbox import distortion_test
# sphinx_gallery_thumbnail_path = '_static/demo_thumbnail/demoImageDistortion.png'

def demo_distortion():
    distortion_test.distortionTest()


if __name__ == "__main__":
    demo_distortion()

###############################################################################
# .. image:: ../_static/demoImageDistortion_1.png
#    :width: 100%

###############################################################################
# .. image:: ../_static/demoImageDistortion_2.png
#    :width: 100%

###############################################################################
# .. image:: ../_static/demoImageDistortion_3.png
#    :width: 100%

###############################################################################
# .. image:: ../_static/demoImageDistortion_4.png
#    :width: 100%