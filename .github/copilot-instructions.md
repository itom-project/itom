# Copilot instructions — itom core

The tool-independent instructions for this repository live in
[`AGENTS.md`](../AGENTS.md). **Read and follow that file.**

Quick reminders:

- C++/Qt, namespace `ito`, error handling via `ito::RetVal` — no exceptions across
  DLL boundaries.
- `ito::DataObject` is stored plane-wise and may carry a ROI; use `getNumPlanes()` /
  `getCvPlaneMat(i)` / `getSize(dim)`.
- Python bindings in `Qitom/python/` must follow the existing CPython reference-counting
  and error-reporting patterns; new bindings need doc strings and `docs/` entries.
- Do not break the public API — out-of-tree plugins depend on it.
- Formatting via `.clang-format` (Microsoft style, 4 spaces, 100 columns); format only
  the lines you touch.
- Never edit generated files or anything inside the CMake binary directory.
