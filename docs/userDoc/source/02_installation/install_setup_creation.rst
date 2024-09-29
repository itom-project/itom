.. include:: ../include/global.inc

.. |star| unicode:: U+002A

Setup creation
**************

Responsible for Setup Releases and maintenance of the versioning system is
the [ITOM Release-Team](https://github.com/orgs/itom-project/teams/itom-release/).

Version Management in ITOM
--------------------------

ITOM version numbers are assigned and maintained via Git-Tags.
For each Release a specific GIT Tag is created with regards to
the central **itomProject** repository and it's correspondand
submodules (e.g. **itom**, **plugins** and **designerPlugins**).

Git TAG are assigned according to the principle of [semantic versioning](https://semver.org)
according to the syntax:

**v\<MAJOR\>.\<MINOR\>.\<PATCH\>[-\<IDENTIFIERS\>]**

1. \<MAJOR\> numeric value changes when a new Release is incompatible with the previous API
2. \<MINOR\> numeric value changes backward compatible functionality is added
3. \<PATCH\> numeric value changes backward compatible bug fixes are made
4. \<IDENTIFIERS\> additional string designator to highlight non-release Tags not intended
   for official releases and official distributions


CMake automatically detects the latest Tag number in a series of derived branches.
It sets the version number for the following files in the build directory:

- itom\Qitom\global.h
- itom\Qitom\version.rc
- itom\itomWidgets\global.h
- itom\SDK.h
- plugins\<PluginName>\pluginVersion.h
- designerplugins\<PluginName>\pluginVersion.h

> Note: It is possible for Developers to combine different versions of Itom, Plugins and
> Designerplugins, by checking out different sets of Git-Tags.
> This is not intended for regular releases, whereby the version number should be
> unified throughout all Repositories.

Besides this the Itom-Release team checks the manually assigned Version Number
for the Interface modules in the files **addInInterfaceVersion.h** and
**designerPluginInterfaceVersion.h**. They are not related to the Itom Versioning
system.


Create Setup for Windows
------------------------

1. Download the latest `InnoSetup <https://jrsoftware.org/isinfo.php>`_ .
2. Build the Documentation according to :ref:`build-documentation-label` section.
3. Run the "start_qt_deployment.bat" file in "[itomProject]\build\itom\setup\win64"
4. Download the mandatory and optional Python wheels:
  - open the command line interface
  - got to "[itomProject]\build\itom\setup\win64\PythonRequirements"
  - run: "python -m pip download --prefer-binary -r requirementsMandatory.txt"
  - run: "python -m pip download --prefer-binary -r requirementsOptional.txt"
5. To create a Windows Setup install open the file build\itom\setup\win64\itom_setup_win64.iss in the InnoSetupCompiler and run it accordingly.
