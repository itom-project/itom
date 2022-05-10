"""Roughness evaluation
=======================

This demo opens the Roughness Evaluator app.

This app is contained in the apps.roughnessEvaluator module
of the itom-packages folder of ``itom``. 
The usage of this demo requires the algorithm plugin ``Roughness``.
"""

from apps.roughnessEvaluator import profile_roughness  # from itom-packages
from itom import pluginLoaded
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoRoughnessEvaluator.png'

if pluginLoaded("Roughness") == False:
    raise RuntimeError("The algorithm plugin 'Roughness' is not available")


profile_roughness = profile_roughness.ProfileRoughness()
profile_roughness.show()


###############################################################################
# .. image:: ../../_static/demoRoughnessEvaluator_1.png
#    :width: 100%