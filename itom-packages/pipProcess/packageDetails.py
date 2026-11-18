"""Helper module to get the details of all installed Python packages in one call.

This module is called from pipManager (Python Package Manager of itom) as a fast
alternative to ``python -m pip show <packages>``. ``pip show`` loads the full pip
machinery for every call and computes the "Required-by" field, which parses the
requirements of all installed packages for every requested package. Both is very
slow for many installed packages. This script only uses ``importlib.metadata``.

The result is printed to stdout as one JSON list (ASCII only, so the stream encoding
of the calling QProcess does not matter). Every item is a dict with the keys
``name``, ``version``, ``location``, ``requires``, ``summary``, ``homepage`` and
``license``. The values are built like the corresponding fields of ``pip show``.

Exit codes: 0 - success, 1 - any error (the caller then falls back to ``pip show``).
"""

import json
import re
import string
import sys

try:
    import importlib.metadata as importlib_metadata
except ImportError:  # Python < 3.8
    sys.stderr.write("importlib.metadata is not available.\n")
    sys.exit(1)

try:
    from packaging.requirements import Requirement
except ImportError:
    try:
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        Requirement = None

_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _canonical_name(name):
    """Canonical package name (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requires(dist):
    """Names of the requirements without extras, sorted like ``pip show``."""
    names = set()

    for req_string in dist.requires or []:
        req_string = req_string.strip()

        if Requirement is not None:
            try:
                req = Requirement(req_string)
            except Exception:
                req = None

            if req is not None:
                try:
                    if req.marker is None or req.marker.evaluate({"extra": ""}):
                        names.add(req.name)
                except Exception:
                    names.add(req.name)
                continue

        # fallback without packaging: skip requirements that only belong to an extra
        name_and_marker = req_string.split(";", 1)

        if len(name_and_marker) == 2 and "extra" in name_and_marker[1]:
            continue

        match = _NAME_PATTERN.match(name_and_marker[0])

        if match:
            names.add(match.group(1))

    return sorted(names, key=str.lower)


def _homepage(metadata):
    """Home-page, or the 'homepage' Project-URL (like ``pip show``)."""
    homepage = metadata.get("Home-page") or ""

    if not homepage:
        removal_map = str.maketrans("", "", string.punctuation + string.whitespace)

        for url in metadata.get_all("Project-URL") or []:
            label, sep, value = url.partition(",")

            if sep and label.translate(removal_map).lower() == "homepage":
                homepage = value.strip()
                break

    return homepage


def _first_line(text):
    """``pip show`` writes multi-line fields over several lines, but itom only uses the first one."""
    return (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""


def _location(dist):
    """Directory that contains the .dist-info / .egg-info folder."""
    try:
        return str(dist.locate_file(""))
    except Exception:
        return ""


def main():
    packages = []
    found = set()

    for dist in importlib_metadata.distributions():
        metadata = dist.metadata
        name = metadata["Name"] if metadata is not None else None

        if not name:
            continue

        # the first distribution of a name on sys.path is the one that Python imports
        key = _canonical_name(name)

        if key in found:
            continue

        found.add(key)

        packages.append(
            {
                "name": name,
                "version": metadata["Version"] or "",
                "location": _location(dist),
                "requires": ", ".join(_requires(dist)),
                "summary": _first_line(metadata.get("Summary")),
                "homepage": _homepage(metadata),
                "license": _first_line(
                    metadata.get("License-Expression") or metadata.get("License")
                ),
            }
        )

    sys.stdout.write(json.dumps(packages, ensure_ascii=True))
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stderr.write("Error getting package details: %s\n" % ex)
        sys.exit(1)
