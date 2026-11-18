#!/usr/bin/env python3
"""Runs the GoogleTest executables of itom and compares the failing tests with a list of known failures.

Why this script exists:
    The main() functions of the unittest executables always return 0 (and wait for a key press, unless
    the argument -googletestadapter is given). The exit code can therefore not be used to detect failing
    tests. This script runs every executable with the JSON output of GoogleTest and evaluates it.

Result:
    - a failing test, that is NOT listed in the file of known failures, fails the run (exit code 1),
    - a crashing executable (no result file) fails the run,
    - a known failure that passes now is reported as a notice. Remove it from the list of known
      failures (or run this script with --update-known).

Usage:
    python run_unittests.py --build-dir build --known known_failures_linux.txt
    python run_unittests.py --build-dir build --known known_failures_linux.txt --update-known

Format of the file of known failures (one test per line, '#' starts a comment):
    unittest_dataobject: dataObjectTagSpace_operator_Test/0.tagSpace_comp_Test
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def find_executables(build_dir, names):
    suffix = ".exe" if os.name == "nt" else ""
    if names:
        return [os.path.join(build_dir, n + suffix) for n in names]

    result = []
    for entry in sorted(os.listdir(build_dir)):
        path = os.path.join(build_dir, entry)
        if (
            entry.startswith("unittest_")
            and os.path.isfile(path)
            and (entry.endswith(".exe") if os.name == "nt" else os.access(path, os.X_OK) and "." not in entry)
        ):
            result.append(path)
    return result


def read_known(path):
    known = {}
    if not path:
        return known
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            exe, _, test = line.partition(":")
            known.setdefault(exe.strip(), set()).add(test.strip())
    return known


def run_executable(exe, workdir, timeout, extra_args):
    """returns (returncode, result dict or None)."""
    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "result.json")
        cmd = [exe, "-googletestadapter", "--gtest_output=json:" + json_path] + extra_args
        env = dict(os.environ)
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            proc = subprocess.run(cmd, cwd=workdir, env=env, timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            return "timeout", None
        except OSError as e:
            return "cannot start (%s)" % e, None

        if not os.path.isfile(json_path):
            return returncode, None
        with open(json_path, encoding="utf-8") as f:
            return returncode, json.load(f)


def count_skipped(result):
    return sum(
        1
        for suite in result.get("testsuites", [])
        for test in suite.get("testsuite", [])
        if test.get("result") == "SKIPPED"
    )


def failing_tests(result):
    failing = set()
    for suite in result.get("testsuites", []):
        for test in suite.get("testsuite", []):
            if test.get("failures"):
                failing.add("{}.{}".format(suite["name"], test["name"]))
    return failing


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", required=True, help="folder with the unittest executables")
    parser.add_argument("--known", help="file with the known failures")
    parser.add_argument("--exe", action="append", help="name of an executable (default: all unittest_*)")
    parser.add_argument("--timeout", type=int, default=900, help="timeout per executable in seconds")
    parser.add_argument("--update-known", action="store_true", help="write the current failures to the --known file")
    parser.add_argument("--json-dir", help="folder to store the result files of GoogleTest")
    parser.add_argument("gtest_args", nargs="*", help="further arguments for the executables (after --)")
    args = parser.parse_args()

    # The executables are started with the build directory as working directory. A relative path would then
    # be resolved a second time relative to it ('build/build/unittest_x').
    build_dir = os.path.abspath(args.build_dir)

    known = read_known(args.known) if not args.update_known else {}
    executables = find_executables(build_dir, args.exe)
    if not executables:
        print("No unittest executables found in %s" % build_dir)
        return 1

    ok = True
    current = {}
    rows = []

    for exe in executables:
        name = os.path.splitext(os.path.basename(exe))[0]
        print("\n===== %s =====" % name, flush=True)
        if not os.path.isfile(exe):
            print("::error::%s was not built" % name)
            ok = False
            rows.append((name, "-", "-", "-", "-", "not built"))
            continue

        returncode, result = run_executable(exe, build_dir, args.timeout, args.gtest_args)

        if result is None:
            print(f"::error::{name} crashed or timed out (return code {returncode}), no test result available")
            ok = False
            rows.append((name, "-", "-", "-", "-", "CRASH (%s)" % returncode))
            continue

        if args.json_dir:
            os.makedirs(args.json_dir, exist_ok=True)
            with open(os.path.join(args.json_dir, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(result, f)

        failing = failing_tests(result)
        current[name] = failing
        expected = known.get(name, set())
        new = sorted(failing - expected)
        fixed = sorted(expected - failing)
        total = result.get("tests", 0)

        if total == 0:
            print(
                "::error::%s ran no tests. Are tests compiled out by a missing CMake option "
                "(e.g. BUILD_UNITTEST_INTERNAL_ENABLE=ON)?" % name
            )
            ok = False

        for t in new:
            print(f"::error::{name}: {t} failed (not a known failure)")
        for t in fixed:
            print(f"::notice::{name}: {t} is listed as known failure, but passes now. Remove it from the list.")

        if new:
            ok = False
        skipped = count_skipped(result)
        if skipped:
            print("::notice::%s: %d test(s) skipped (see the output above)" % (name, skipped))
        status = "NO TESTS" if total == 0 else ("FAILED" if new else "ok")
        rows.append((name, total, skipped, len(failing) - len(new), len(new), status))

    if args.update_known:
        with open(args.known, "w", encoding="utf-8") as f:
            f.write("# Known failing tests, see run_unittests.py. Fix the tests and remove them from this list.\n")
            for exe in sorted(current):
                for t in sorted(current[exe]):
                    f.write(f"{exe}: {t}\n")
        print("\nWrote %s" % args.known)

    summary = [
        "| Executable | Tests | Skipped | Known failures | New failures | Result |",
        "|---|---|---|---|---|---|",
    ]
    summary += ["| %s | %s | %s | %s | %s | %s |" % r for r in rows]
    print("\n" + "\n".join(summary))

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write("### Unit tests\n\n" + "\n".join(summary) + "\n")

    return 0 if (ok or args.update_known) else 1


if __name__ == "__main__":
    sys.exit(main())
