.. _translations:

Translations
************************

General
============

The basic language of itom or its plugins is English. All strings in the source code should be
written in English.

itom and its plugins can be translated to different other languages using the `translation system
of Qt <https://doc.qt.io/qt-5/i18n-source-translation.html>`.

The source translation strings of the core project of itom, its libraries (part of the itom SDK), designer plugins or
plugins are stored in different ts files whose filename should follow the pattern: **libraryname_langid.ts**,
where **libraryname** is a placeholder for the corresponding library or plugin and **langid** is a
placeholder for the particular language.

The language can have the following format:

* **de** for German
* **de_DE** for German / Germany
* **de_CH** for German / Switzerland
* **ko** for Korean

For language codes (small letters) and / or country codes (capital letters)
see https://docs.oracle.com/cd/E13214_01/wli/docs92/xref/xqisocodes.html.

Usually the source strings (in English) of the **ts** files are automatically created or updated by Qt's tool **lupdate**. This tool
parses all corresponding source files (h, cpp, ui), detects all translatable strings and updates the ts files.

The translations itself are then comfortabily done in the tool QtLinguist of Qt.

The **ts** files are then compiled using Qt's **lrelease** tool into their binary representation **qm**. Each
**qm** file has the same base filename than its original **ts** file and is put in several **translation** subfolders
of the install or build folder of itom or its plugins subfolder. This qm-file compilation is automatically
done during each build of the corresponding library.

For starting itom with another language, choose the desired language in the :ref:`property dialog of itom <gui-propertydialog-language>`.
Possible languages depend on the available **qm** translation files:

Adding translations for the itom core project
====================================================

Adding new or existing translations for different languages to the itom core project or some
of its plugins or designerplugins is usually done
in the CMake configuration process.

Select the desired language codes (see above) as semicolon-separated string to the CMake variable
**ITOM_LANGUAGES**. Since English is the default language, never add **en** to this string.

Example:

.. code-block:: cmake

    ITOM_LANGUAGES = de;fr
    #this includes translations for German and French (in addition to English)

If you press **Configure** in CMake after having changed the **ITOM_LANGUAGES** variable, CMake checks
if the corresponding **ts** files are available for all desired languages and all translatable libraries, that
belong to the itom core project.

There is one common CMake error message, that might occur when setting new languages:

Error (ITOM_LANGUAGES = it):

.. code-block:: cmake

    CMake Error at cmake/ItomBuildMacros.cmake:664 (MESSAGE):
      Source translation file
      'C:/itom/sources/itom/Qitom/translation/qitom_it.ts' for language 'it is
      missing.  Please create this file first or set ITOM_UPDATE_TRANSLATIONS to
      ON
    Call Stack (most recent call first):
      Qitom/CMakeLists.txt:759 (itom_library_translation)

CMake could not find the relevant Italian translation file **qitom_it.ts**. Since the additional CMake
variable **ITOM_UPDATE_TRANSLATIONS** is likely to be set to **OFF**, itom is not allowed to automatically
create an initial version of this file, such that the error is raised.

**Solution:** Set **ITOM_UPDATE_TRANSLATIONS** to **ON** if you add new languages. Then a bare version of
a **ts** file is created for each non-existing language and Qt's **lupdate** process is triggered when
building each library. This also affects all existing **ts** file, such that they are also updated by new
translatable strings in the source code.

.. note::

    It is not recommended to manually create a new ts-file for a new language by copying an existing ts-file
    from another language, since each ts-file contains xml-content with the language ID as subcontent. If this
    language ID does not fit to the suffix of the filename, the **lupdate** process to update the translatable
    strings in this file will fail.


Updating or creating new ts-files
====================================

Usually, the itom build process does not influence any ts-files during the build, hence, no-existing ts-files
are not updated nor new ts-files for new languages in the CMake variable **ITOM_LANGAUGES** are created. However
the binary compilation of qm-files from existing ts-files is always started when building the particular
library.

The source translation update process can be triggered and controlled by the two boolean CMake variables
**ITOM_UPDATE_TRANSLATIONS** and **ITOM_UPDATE_TRANSLATIONS_REMOVE_UNUSED_STRINGS**.

Set **ITOM_UPDATE_TRANSLATIONS** to force the itom (or plugins) build process to always update or
create the **ts**-files for all desired languages (based on **ITOM_LANGAUGES**) using the tool
**lupdate** of Qt. This is done by a special build project, whose name is equal to the corresponding
library, followed by the suffix **_translation** (e.g. **itomCommonQtLib_translation** for the library
**itomCommonQtLib**. The original library project depends on its *_translation*-project, such that this
is always created before the library itself.

Usually, the **lupdate** tool parses all header, source and user-interface files of the library for translatable
strings and adds them to the **ts** file, if they do not exist yet. However **lupdate** does not remove unused
strings from the **ts** file. This can be changed by also setting **ITOM_UPDATE_TRANSLATIONS_REMOVE_UNUSED_STRINGS** to
**ON**.

Please commit all changed **ts** files to the Git repository of the corresponding library (or plugin).

If **ITOM_UPDATE_TRANSLATIONS** is **ON** and a **ts** file has a wrong format, the following CMake
error might occur:

.. code-block:: cmake

    CMake Warning at cmake/ItomBuildMacros.cmake:745 (message):
      - The existing ts-file
      C:/itom/sources/itom/AddInManager/translation/addinmanager_fr.ts does not
      contain the required language 'fr', but 'de'.  The lupdate process might
      fail.  Either fix the file or delete it and re-configure to let CMake
      rebuild an empty, proper ts file.
    Call Stack (most recent call first):
      cmake/ItomBuildMacros.cmake:656 (itom_qt5_create_translation)
      AddInManager/CMakeLists.txt:96 (itom_library_translation)

In this case, the translation system expected a french translation file **addinmanager_fr.ts**, however the
file was a copy from the german version **addinmanager_de.ts**, such that the internal xml content of the
file was invalid. If this happens, **lupdate** will fail later. To fix this, remove the file and run CMake
again or fix the content of the file.

Location of ts-files
===========================

Usually the ts-files are always located in a **translation** subfolder of the sources
of the wrapped library. These are for the itom core project:

* sources/itom/AddInManager/translation
* sources/itom/itomCommonQt/translation
* sources/itom/itomWidgets/translation
* sources/itom/plot/translation
* sources/itom/shape/translation
* sources/itom/qitom/translation

The translation files for plugins are always located in a subfolder **translation** of
the particular plugin sources. The same holds for designerplugins.

When deploying itom, the qm-files are located in the following folders:

* itom core project: itom-subfolder **translation**
* designer plugins: itom-subfolder **designer/translation**
* plugin (name: targetname): itom-subfolder **plugins/<targetname>/translation**

Translating plugins or designerplugins
=======================================

While the itom core project can directly be translated by setting the mentioned CMake variables
**ITOM_LANGUAGES**, **ITOM_UPDATE_TRANSLATIONS** and optionally **ITOM_UPDATE_TRANSLATIONS_REMOVE_UNUSED_STRINGS**,
some additional steps have to be done for translating plugins and / or designerplugins.

Besides configuring the mentioned CMake variables, you also have to set the following exemplary
lines into the CMakeLists.txt file of the particular plugin:

.. code-block:: cmake

    #translation
    set(FILES_TO_TRANSLATE ${PLUGIN_SOURCES} ${PLUGIN_HEADERS} ${PLUGIN_UI})
    itom_library_translation(QM_FILES TARGET ${target_name} FILES_TO_TRANSLATE ${FILES_TO_TRANSLATE})

The **FILES_TO_TRANSLATE** list will contain all source (cpp), header (h) or user interface (ui) files,
that should be parsed for new files. The translation project itself will be created by the itom macro
**itom_library_translation**, provided by **ItomBuildMacros.cmake**. This will then create ts
files in a **translation** subfolder of the plugin source folder whose particular names are **targetname_langid.ts**,
where **targetname** corresponds to the project name of the plugin and **langid** is each language ID, contained
in the semicolon-separated list **ITOM_LANGUAGES**.
