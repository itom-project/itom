"""Tests of run_unittests.py with small fake test executables (no build of itom required).

Run with:  python3 -m unittest discover -s itom_unittests/ci -p "test_*.py"
"""

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_unittests.py")

FAKE_EXE = """#!{python}
import json, sys
out = [a.split(":", 1)[1] for a in sys.argv if a.startswith("--gtest_output=json:")][0]
assert "-googletestadapter" in sys.argv
tests = {tests}
suite = [{{"name": n, "result": r, **({{"failures": [{{"failure": "x"}}]}} if r == "FAILED" else {{}})}} for n, r in tests]
json.dump({{"tests": len(tests), "testsuites": [{{"name": "Suite", "testsuite": suite}}]}}, open(out, "w"))
"""

CRASHING_EXE = "#!/bin/sh\nkill -SEGV $$\n"


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.build = os.path.join(self.root, "build")
        os.mkdir(self.build)

    def tearDown(self):
        self._tmp.cleanup()

    def add_exe(self, name, tests=None, content=None):
        path = os.path.join(self.build, name)
        if content is None:
            content = FAKE_EXE.format(python=sys.executable, tests=repr(tests))
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)

    def known(self, *lines):
        path = os.path.join(self.root, "known.txt")
        with open(path, "w") as f:
            f.write("# comment\n" + "\n".join(lines) + "\n")
        return path

    def run_runner(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, RUNNER] + list(args),
            cwd=cwd or self.root,
            capture_output=True,
            text=True,
            env=dict(os.environ, GITHUB_STEP_SUMMARY=""),
        )

    def test_relative_build_directory(self):
        # as in the workflow: 'python3 .../run_unittests.py --build-dir build', started in the parent folder
        self.add_exe("unittest_a", [("Ok", "COMPLETED")])
        result = self.run_runner("--build-dir", "build", "--known", self.known(), cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_new_failure_fails_the_run(self):
        self.add_exe("unittest_a", [("Ok", "COMPLETED"), ("Bad", "FAILED")])
        result = self.run_runner("--build-dir", self.build, "--known", self.known())
        self.assertEqual(result.returncode, 1)
        self.assertIn("Suite.Bad failed (not a known failure)", result.stdout)

    def test_known_failure_is_accepted(self):
        self.add_exe("unittest_a", [("Ok", "COMPLETED"), ("Bad", "FAILED")])
        result = self.run_runner("--build-dir", self.build, "--known", self.known("unittest_a: Suite.Bad"))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_known_failure_of_another_executable_does_not_count(self):
        self.add_exe("unittest_a", [("Bad", "FAILED")])
        result = self.run_runner("--build-dir", self.build, "--known", self.known("unittest_b: Suite.Bad"))
        self.assertEqual(result.returncode, 1)

    def test_fixed_known_failure_gives_a_notice(self):
        self.add_exe("unittest_a", [("Ok", "COMPLETED")])
        result = self.run_runner("--build-dir", self.build, "--known", self.known("unittest_a: Suite.Ok"))
        self.assertEqual(result.returncode, 0)
        self.assertIn("::notice::unittest_a: Suite.Ok is listed as known failure", result.stdout)

    def test_skipped_tests_are_reported(self):
        self.add_exe("unittest_a", [("Ok", "COMPLETED"), ("Skipped", "SKIPPED")])
        result = self.run_runner("--build-dir", self.build, "--known", self.known())
        self.assertEqual(result.returncode, 0)
        self.assertIn("1 test(s) skipped", result.stdout)

    def test_crash_fails_the_run(self):
        self.add_exe("unittest_crash", content=CRASHING_EXE)
        result = self.run_runner("--build-dir", self.build, "--known", self.known())
        self.assertEqual(result.returncode, 1)
        self.assertIn("crashed", result.stdout)

    def test_missing_executable_fails_the_run_without_traceback(self):
        result = self.run_runner("--build-dir", self.build, "--exe", "unittest_missing", "--known", self.known())
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)

    def test_not_executable_file_fails_the_run_without_traceback(self):
        self.add_exe("unittest_a", content="not a program")
        os.chmod(os.path.join(self.build, "unittest_a"), 0o755)
        result = self.run_runner("--build-dir", self.build, "--exe", "unittest_a", "--known", self.known())
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)

    def test_executable_without_tests_fails_the_run(self):
        # e.g. all tests are compiled out, because a CMake option is missing
        self.add_exe("unittest_a", [])
        result = self.run_runner("--build-dir", self.build, "--known", self.known())
        self.assertEqual(result.returncode, 1)
        self.assertIn("ran no tests", result.stdout)
        self.assertIn("| unittest_a | 0 | 0 | 0 | 0 | NO TESTS |", result.stdout)

    def test_no_executables_fails_the_run(self):
        result = self.run_runner("--build-dir", self.build, "--known", self.known())
        self.assertEqual(result.returncode, 1)

    def test_json_results_are_stored(self):
        self.add_exe("unittest_a", [("Ok", "COMPLETED")])
        out = os.path.join(self.root, "results")
        result = self.run_runner("--build-dir", self.build, "--known", self.known(), "--json-dir", "results")
        self.assertEqual(result.returncode, 0)
        with open(os.path.join(out, "unittest_a.json")) as f:
            self.assertEqual(json.load(f)["tests"], 1)


if __name__ == "__main__":
    unittest.main()
