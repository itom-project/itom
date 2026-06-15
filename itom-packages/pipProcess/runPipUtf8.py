# coding=utf8

"""Helper module to call pip from a QProcess and forcing the stdout and sterr streams to UTF-8.

This module is called from pipManager as an alternative to the direct call of pip.
It should only be used under Windows, since QProcess establishes out and err streams
using the cp1252 encoding. This encoding might result in UnicodeEncodeErrors, if
for instance a pip package contains an author name with special characters.
"""

import sys
import runpy

# get arguments, the first argument must be 'pip'
args = sys.argv[1:]

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# for the following code, see the comment in
# pip/_internal/cli/main.py
sys.argv = args
runpy.run_module("pip", run_name="__main__")
