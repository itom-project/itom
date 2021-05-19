"""This demo opens the Roughness Evaluator app.

This app is contained in the apps.roughnessEvaluator module
of the itom-packages folder of itom.

The usage of this demo requires the algorithm plugin 'Roughness'.
"""

if pluginLoaded("Roughness") == False:
    raise RuntimeError("The algorithm plugin 'Roughness' is not available")

from apps.roughnessEvaluator import profile_roughness  # from itom-packages

profile_roughness = profile_roughness.ProfileRoughness()
profile_roughness.show()
