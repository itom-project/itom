"""Tests of check_exported_symbols.py with a small shared library built on the fly (Linux, g++ required).

Run with:  python3 -m unittest discover -s itom_unittests/ci -p "test_*.py"
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest

CHECKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_exported_symbols.py")

LIB_SOURCE = """
#include <string>
namespace ito {
class Foo {
public:
    int strongMethod() const;
    template <typename T> T weakMethod(T v) const { return v; }
};
int Foo::strongMethod() const { return 1; }
template int Foo::weakMethod<int>(int) const;
int freeFunction() { return 2; }
%s
}
extern "C" int c_function() { return 3; }
"""

REMOVED_METHOD = "int Foo_extra() { return 4; }"


@unittest.skipUnless(platform.system() == "Linux" and shutil.which("g++") and shutil.which("nm"), "needs Linux, g++, nm")
class CheckExportedSymbolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.build = os.path.join(self.root, "build")
        self.reference = os.path.join(self.root, "abi")
        os.mkdir(self.build)

    def tearDown(self):
        self._tmp.cleanup()

    def build_lib(self, extra=""):
        src = os.path.join(self.root, "lib.cpp")
        with open(src, "w") as f:
            f.write(LIB_SOURCE % extra)
        subprocess.run(
            ["g++", "-shared", "-fPIC", "-o", os.path.join(self.build, "libdataobject.so"), src],
            check=True,
        )

    def run_checker(self, *args):
        return subprocess.run(
            [sys.executable, CHECKER, "--build-dir", self.build, "--reference", self.reference, "--lib", "dataobject"]
            + list(args),
            capture_output=True,
            text=True,
            env=dict(os.environ, GITHUB_STEP_SUMMARY=""),
        )

    def reference_file(self):
        return os.path.join(self.reference, "dataobject.linux-%s.txt" % platform.machine().lower())

    def test_update_writes_reference_and_check_passes(self):
        self.build_lib()
        result = self.run_checker("--update")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with open(self.reference_file()) as f:
            content = f.read()
        self.assertIn("strong _ZNK3ito3Foo12strongMethodEv", content)
        self.assertIn("strong c_function", content)
        self.assertIn("weak _ZNK3ito3Foo10weakMethodIiEET_S2_", content)
        self.assertNotIn("_ZNSt", content)

        result = self.run_checker()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("| dataobject |", result.stdout)
        self.assertIn("| ok |", result.stdout)

    def test_removed_strong_symbol_fails(self):
        self.build_lib(REMOVED_METHOD)
        self.run_checker("--update")
        self.build_lib()
        result = self.run_checker()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("::error::", result.stdout)
        self.assertIn("ito::Foo_extra()", result.stdout)

    def test_new_symbol_is_only_a_notice(self):
        self.build_lib()
        self.run_checker("--update")
        self.build_lib(REMOVED_METHOD)
        result = self.run_checker()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("::notice::dataobject: new exported symbol", result.stdout)

    def test_missing_weak_symbol_is_only_a_notice(self):
        self.build_lib()
        self.run_checker("--update")
        with open(self.reference_file(), "a") as f:
            f.write("weak _ZNK3ito3Foo10weakMethodIdEET_S2_\n")
        result = self.run_checker()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("::notice::dataobject: weak symbol missing", result.stdout)

    def test_missing_reference_fails(self):
        self.build_lib()
        result = self.run_checker()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("no reference file", result.stdout)
