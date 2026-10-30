.. include:: ../include/global.inc

.. |mm| unicode:: U+00B5 m

Changelog Versions 5.x
######################

Version 5.0.0 (2025-MM-DD)
**************************

Itom-Project
------------

* Added external git submodule

External
--------
 * This submodule comprises a CMake project to build a relevant external dependencies
 such as Qt, OpenCV Boost, VTK and PCL for all platforms from within the itom-project.
 This is referred as ExternalProject and aims to overcome in particular the binary
 dependencies on Windows Platforms and should enable homogeneous versions and code dependencies
 across all platforms.

itom
----

* The script editor can now open and edit other text-like files (e.g. txt, json, yaml, xml, ini, md, cmake, bat) besides python scripts.
  These files get a suffix specific syntax highlighting. Python specific features (run, debug, code checker, auto completion, ...) are disabled for them.
  The list of supported suffixes is configurable in the property dialog (tab scripts). See :ref:`gui-editor-text-files`.
* ``itom.openScript`` can also open the supported text files in the script editor.
* The script editor detects UTF-16 and UTF-32 files with a byte order mark (e.g. data listings exported by Windows tools) and loads them correctly. The BOM of UTF-8/16/32 files is written again when the file is saved (before, it was dropped).
* Fix: files were read and written in text mode, which removed or added ``\r`` bytes before the decoding. UTF-16/32 files were corrupted (garbage characters, missing lines) and the garbage made the editor very slow. Files are now read and written in binary mode, the line endings are handled after the decoding.
* The python specific actions (run, debug, step, auto format, rename, docstring), the script toolbar and the outline navigation bar are hidden, if the current tab is a text file.
* Script editor: very long lines (e.g. data listings with thousands of characters per line) are drawn clipped to the visible part. Before, Qt drew the complete lines at every repaint, such that every keystroke took hundreds of milliseconds in such files.
* Script editor: the highlighting of the occurrences of the word under the cursor was very slow (seconds) for frequent words in big files. The search is now linear and stops at the maximum number of highlighted occurrences.
* New unittest ``unittest_codeeditor`` (GoogleTest) for the file type registry and the syntax highlighting of the text-like files.
* Fix: itom crashed at startup if the clipboard has no mime data (e.g. with the Qt platform plugin ``offscreen``).
* Script editor windows can be docked and undocked by drag and drop. Tabs can be dragged to other script windows, to the edges of the main window or outside of itom. Undocked script windows can be docked by releasing them above the edges of the main window. See :ref:`gui-editor-dock-drag-drop`.
