# AGENTS.md — itom core

Guidance for AI coding agents working in the **itom core** repository.

This repository is normally used as a submodule of `itomProject`. If the umbrella
repository is checked out, its `AGENTS.md` applies in addition to this file.

## Architecture

| Folder             | Purpose                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `DataObject`       | `ito::DataObject` — n-dimensional array type (plane-wise storage, ROI, axis/tag meta data) |
| `PointCloud`       | point cloud and polygon mesh wrappers (PCL, optional)                   |
| `shape`            | `ito::Shape` geometric primitives                                        |
| `common`           | non-Qt basics: `RetVal`, `Param`/`ParamMeta`, `AddInInterface`, type defs |
| `commonQt`         | Qt-dependent helpers, `apiFunctions`, plugin API                        |
| `AddInManager`     | discovery, loading and lifetime management of plugins                   |
| `Qitom`            | application, main window, script editor, **embedded Python** (`Qitom/python`) |
| `itomWidgets`      | reusable Qt widgets                                                     |
| `QPropertyEditor`  | property grid used by the GUI                                           |
| `plot`             | plot interfaces (`AbstractFigure`, `AbstractDObjFigure`, ...)            |
| `pluginTemplates`  | skeletons for new plugins                                               |
| `itom_unittests`   | GoogleTest based unit tests                                             |
| `python_unittests` | Python level tests, executed inside itom                                |
| `docs`             | Sphinx documentation                                                    |

## Hard rules

1. **Do not break the public API.** Out-of-tree plugins link against `itomCommonLib`,
   `itomCommonQtLib`, `dataobject`, `pointcloud` and the plot interfaces. Add new
   overloads, deprecate old ones, never silently change signatures or `enum` values.
2. **Error handling via `ito::RetVal`**, not exceptions. Accumulate with `+=`, check
   with `containsError()` / `containsWarningOrError()`. Exceptions must not cross a
   DLL/plugin boundary.
3. **`DataObject` is plane-wise and may have a ROI.** Use `getNumPlanes()` +
   `getCvPlaneMat(i)` and `getSize(dim)`; only use raw pointers after checking
   `getContinuous()`.
4. **Python bindings** (`Qitom/python/*.cpp`) must follow the existing CPython
   reference-counting, `PyMethodDef`/`PyGetSetDef` and doc-string patterns. Every new
   binding needs documentation in `docs/`.
5. **Threading:** GUI objects are only touched in the main thread; plugin and Python
   threads communicate via signals/slots or `QMetaObject::invokeMethod`.
6. **Translations:** user visible strings go through `tr(...)`; do not add new
   hard-coded strings to the GUI.

## Style

- `.clang-format` (`BasedOnStyle: Microsoft`, 4 spaces, `ColumnLimit: 100`,
  `PointerAlignment: Left`, `AlignAfterOpenBracket: AlwaysBreak`). Format only the lines
  you change.
- Namespace `ito` for public types; `m_` prefix for members.
- English identifiers and comments; Doxygen comments where the file already uses them.

## Build and test

```powershell
cmake -S <sources>/itomProject -B <build>/itomProject
cmake --build <build>/itomProject --config Debug --target unittest_dataobject
```

Never edit files in the binary directory or generated `moc_*`/`ui_*`/`qrc_*` files.
Run the unit test target that matches the changed area (`unittest_dataobject`,
`unittest_commonlib`, `unittest_commonqtlib`, `unittest_addinmanager`,
`unittest_qpropertyeditor`).

## Quality gates

`pre-commit` runs `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`,
`fix-byte-order-marker`, `codespell`, `pyupgrade --py36-plus` and `sphinx-lint`.
Files end with a single newline, contain no trailing whitespace and no BOM.

## Used 3rd party libraries

itom or its plugins are using different 3rd party libraries. If a library from the following list
is used, make sure that the code is compatible to the indicated version range:

* CMake, 3.12 - 4.4, if possible even newer versions
* Qt, 5.11 - 6.11, Qt6 is preferred
* OpenCV, version 3.0 or higher (up to 5.x)
* Python, version 3.6 or higher (3.11 or higher preferred)
* Numpy, version 1.x and 2.x
* PointCloudLibrary, 1.5 - 1.15

It is allowed to use compiler pre-processors to exclude features if a specific, supported, version of the
3rd party library does not support this feature.

For Qt5, the minimum C++ compiler is C++11. For Qt6, the minimum C++ compiler is C++17.

