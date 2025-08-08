"""Execute all example scripts in docs/*_src"""

import subprocess
import sys
from pathlib import Path

docdir = Path(__file__).parent / "docs"

fails = []

for srcdir in sorted(docdir.glob("*_src")):
    print(80 * "=")
    print(f"Running scripts in {srcdir.stem}")
    print(80 * "=")

    for srcfile in sorted(srcdir.glob("*.py")):
        if srcfile.name.startswith("_"):
            continue

        name = f"{srcdir.name}/{srcfile.name}"
        print("\n" + 80 * "-")
        print(f"Running {name}")
        print(80 * "-" + "\n")

        proc = subprocess.run(  # noqa: S603
            [sys.executable, srcfile], check=False, env={"MPLBACKEND": "agg"}
        )
        if proc.returncode:
            fails.append(name)

if fails:
    print("\n\n")
    print(80 * "=")
    print("Failed cases:")
    print(*fails, sep="\n")
    print(80 * "=")

    sys.exit(1)
