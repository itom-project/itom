# Continuous integration of the unittests

The workflow `.github/workflows/tests.yml` builds the libraries and the `unittest_*` executables on Linux
(Qt6) and runs them. A second job measures the line coverage with gcov and publishes it as artifact.

## Why a runner script?

The `main()` functions of the unittest executables always return `0` and wait for a key press, unless the
argument `-googletestadapter` is given. The exit code therefore does not show failing tests.
`run_unittests.py` runs every executable with the JSON output of GoogleTest and evaluates it:

| Situation | Result |
|---|---|
| test fails and is not listed in the known failures | run fails |
| executable crashes, times out or was not built | run fails |
| executable ran no tests (e.g. tests compiled out by a missing CMake option) | run fails |
| test is listed as known failure and passes now | notice, the entry should be removed |
| test is skipped | shown in the summary |

The runner has its own tests with fake executables (no build required):
`python3 -m unittest discover -s itom_unittests/ci -p "test_*.py"`. The workflow runs them first.

## Run it locally

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_PCL=OFF -DBUILD_WITH_HELPVIEWER=OFF \
      -DBUILD_UNITTEST=ON -DBUILD_UNITTEST_INTERNAL_ENABLE=ON -DBUILD_QTVERSION=Qt6
cmake --build build --target unittest_dataobject unittest_commonlib unittest_commonqtlib \
      unittest_addinmanager unittest_qpropertyeditor unittest_codeeditor
python3 itom_unittests/ci/run_unittests.py --build-dir build --known itom_unittests/ci/known_failures_linux.txt
```

`BUILD_UNITTEST_INTERNAL_ENABLE=ON` is required. Without it, tests that need internal symbols are compiled
out (all tests of `unittest_qpropertyeditor`), and the executable runs with 0 tests. The runner reports an
executable without tests as error.

## Known failures

`known_failures_linux.txt` lists tests that fail today (format: `executable: Suite.Test`). The list must
only shrink: fix a test, then remove it from the list. A new failing test must be fixed, not added.
`--update-known` rewrites the list from the current results (use it only to create a baseline).

## Not covered yet

* Windows (MSVC) and Qt5 builds. The expected failures can differ per platform, use one list per platform.
* `python_unittests`, which require a running itom (`itom run=<script>`).
* `unittest_codeeditor` contains a performance guard (`editorPerformanceTest.cpp`): repaint, typing and
  occurrence highlighting in a 500 kB data listing with 5400 characters per line must stay below fixed
  limits. A failure means that a feature does O(file size) or O(line length) work per keystroke again.
* `unittest_codeeditor` compiles the core of the script editor (`Qitom/codeEditor`, no python modes) directly
  into the test and needs the Qt platform plugin `offscreen` (set by the runner).
* `AddInManagerTest.General` needs the plugin `DummyMotor` from the separate `plugins` repository and is
  skipped in a build of this repository alone.

## Python C-API check (`check_python_api.py`)

itom embeds CPython and uses its C-API in `Qitom/python/*.cpp` and `Qitom/models/pipManager.cpp`.
Every Python release removes functions that were deprecated two releases earlier. The job
`python-api` of `tests.yml` compiles these files against the headers of Python 3.12, 3.13, 3.13t
(free-threaded) and 3.14 with `-fsyntax-only` (no link, no full build) and fails on removed *and*
deprecated API. Locally:

```sh
cmake -S . -B build ... -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --target qitom_autogen
python3 itom_unittests/ci/check_python_api.py --build-dir build --python /path/to/python3.14 --fail-on-deprecated
```

Version differences of the C-API are hidden in `Qitom/python/pythonCompat.h` (e.g.
`PythonCompat::weakrefGetRef` instead of the deprecated `PyWeakref_GetObject`).

Free-threaded Python (`python3.13t`): itom compiles, but its thread model needs the GIL. The `itom`
module does not declare `Py_MOD_GIL_NOT_USED`, so Python re-enables the GIL when it is imported;
`PythonEngine` verifies this at startup and refuses to start if the GIL is disabled (`PYTHON_GIL=0`).
