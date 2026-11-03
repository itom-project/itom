#!/usr/bin/env python3
"""Compiles the C++ files that use the Python C-API against the headers of a given Python.

Why this script exists:
    itom embeds CPython and uses its C-API in about 30 files (Qitom/python/*.cpp and
    Qitom/models/pipManager.cpp). Every Python release removes some of the functions that were
    deprecated two releases earlier. A full build of itom is slow, but a syntax-only compile of
    these files against the headers of a newer Python takes a few seconds per file and finds:
    - removed functions, macros and struct members (compile errors),
    - deprecated functions (warnings of -Wdeprecated-declarations), which will become errors in
      a later release.

How it works:
    The compile commands of the configured build (compile_commands.json, written by CMake with
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON) are reused. The include directories of the Python and
    numpy of the build are replaced by those of the Python given with --python, and the compiler
    is run with -fsyntax-only. No object file is written, no link step is needed, hence the
    build only needs the configure step and the 'qitom_autogen' target (moc/uic headers).

Usage:
    cmake -S . -B build ... -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build build --target qitom_autogen
    python3 itom_unittests/ci/check_python_api.py --build-dir build --python /path/to/python3.14

    --python must be able to import numpy. The exit code is 1 if a file does not compile or, with
    --fail-on-deprecated, if a deprecated Python function is used.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import sysconfig
import time

# translation units that use the Python C-API
PYTHON_SOURCES = re.compile(r"(/Qitom/python/[^/]+\.cpp|/Qitom/models/pipManager\.cpp)$")

# only deprecations of the Python C-API are of interest, not those of Qt
PYTHON_DEPRECATION = re.compile(r"‘([^’]*(?:Py|_Py|PY)[A-Za-z_0-9]*[^’]*)’ is deprecated")


def python_info(python):
    """returns (include dir, numpy include dir, version string) of the given python executable."""
    code = (
        "import sysconfig, numpy, sys;"
        "print(sysconfig.get_paths()['include']);"
        "print(numpy.get_include());"
        "print(sys.version.split()[0] + (' free-threaded' if sysconfig.get_config_var('Py_GIL_DISABLED') else ''))"
    )
    out = subprocess.run([python, "-c", code], capture_output=True, text=True, check=True).stdout.splitlines()
    return out[0], out[1], out[2]


def rewrite_command(command, python_include, numpy_include, source):
    """replaces the Python/numpy include dirs and makes the command a syntax-only compile."""
    parts = shlex.split(command)
    result = []
    skip_next = False
    for p in parts:
        if skip_next:
            skip_next = False
            continue
        if p in ("-o", "-c"):
            skip_next = p == "-o"
            continue
        if p.startswith("-I") and (re.search(r"include/python3\.\d+", p) or "numpy" in p):
            continue
        if p.endswith(".cpp") or p.endswith(".cpp.o"):
            continue
        result.append(p)
    result += ["-fsyntax-only", "-Wdeprecated-declarations", "-I" + python_include, "-I" + numpy_include, source]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", required=True, help="configured build folder with compile_commands.json")
    parser.add_argument("--python", default=sys.executable, help="python executable whose headers are used")
    parser.add_argument("--fail-on-deprecated", action="store_true", help="exit code 1 if deprecated Python API is used")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel compiler processes")
    parser.add_argument("files", nargs="*", help="only these source file names (default: all)")
    args = parser.parse_args()

    cc_path = os.path.join(args.build_dir, "compile_commands.json")
    if not os.path.isfile(cc_path):
        print("::error::%s not found. Configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON." % cc_path)
        return 1

    python_include, numpy_include, version = python_info(args.python)
    print(f"Python {version}, headers in {python_include}")

    with open(cc_path, encoding="utf-8") as f:
        commands = [c for c in json.load(f) if PYTHON_SOURCES.search(c["file"])]
    if args.files:
        commands = [c for c in commands if os.path.basename(c["file"]) in args.files]
    if not commands:
        print("::error::no Python translation units found in compile_commands.json")
        return 1

    failed = []
    deprecated = {}
    running = []

    def collect(entry):
        name, proc, started = entry
        stderr = proc.communicate()[1]
        errors = [l for l in stderr.splitlines() if "error:" in l]
        found = sorted({m.group(1) for m in PYTHON_DEPRECATION.finditer(stderr)})
        status = "OK" if proc.returncode == 0 else "FAIL"
        print("%-4s %-32s %4.0fs %s" % (status, name, time.time() - started, ", ".join(found)), flush=True)
        for e in errors[:10]:
            print("     " + e)
        if proc.returncode != 0:
            failed.append(name)
        if found:
            deprecated[name] = found

    for c in commands:
        name = os.path.basename(c["file"])
        cmd = rewrite_command(c["command"], python_include, numpy_include, c["file"])
        proc = subprocess.Popen(cmd, cwd=c["directory"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        running.append((name, proc, time.time()))
        if len(running) >= args.jobs:
            collect(running.pop(0))
    while running:
        collect(running.pop(0))

    summary = ["Python %s: %d files, %d failed, %d with deprecated Python API" % (version, len(commands), len(failed), len(deprecated))]
    for name in failed:
        print(f"::error::{name} does not compile with Python {version}")
    for name, found in deprecated.items():
        print("::warning::{} uses deprecated Python API: {}".format(name, ", ".join(found)))
    print("\n" + summary[0])

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write("## Python C-API check\n\n%s\n" % summary[0])
            for name, found in deprecated.items():
                f.write("- `{}`: {}\n".format(name, ", ".join("`%s`" % x for x in found)))

    if failed or (args.fail_on_deprecated and deprecated):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
