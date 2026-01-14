import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run read_snap, fields, powerspec, and angular_powerspec in sequence."
    )
    parser.add_argument("--base-path", required=True, help="Path containing snapdir_###.")
    parser.add_argument("--snap-num", type=int, required=True, help="Snapshot number.")
    parser.add_argument("--part-type", default="PartType1", help="Particle group to read.")
    parser.add_argument("--ngrid", type=int, default=1024, help="Grid size per dimension.")
    parser.add_argument("--mas", default="CIC", help="Mass assignment scheme.")
    parser.add_argument("--threads", type=int, default=1, help="Threads for power spectrum.")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for all steps.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Pylians output where supported.",
    )
    return parser.parse_args()


def _run(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    scripts_dir = repo_root / "python"

    out_dir = Path(args.out_dir)

    read_snap = [
        sys.executable,
        str(scripts_dir / "read_snap.py"),
        "--base-path",
        args.base_path,
        "--snap-num",
        str(args.snap_num),
        "--part-type",
        args.part_type,
        "--out-dir",
        str(out_dir),
    ]
    _run(read_snap)

    fields = [
        sys.executable,
        str(scripts_dir / "fields.py"),
        "--in-dir",
        str(out_dir),
        "--ngrid",
        str(args.ngrid),
        "--mas",
        args.mas,
        "--out-dir",
        str(out_dir),
    ]
    if args.verbose:
        fields.append("--verbose")
    _run(fields)

    powerspec = [
        sys.executable,
        str(scripts_dir / "powerspec.py"),
        "--in-dir",
        str(out_dir),
        "--ngrid",
        str(args.ngrid),
        "--mas",
        args.mas,
        "--threads",
        str(args.threads),
        "--out-dir",
        str(out_dir),
    ]
    if args.verbose:
        powerspec.extend(["--verbose", "True"])
    _run(powerspec)

    angular = [
        sys.executable,
        str(scripts_dir / "angular_powerspec.py"),
        "--in-dir",
        str(out_dir),
        "--out-dir",
        str(out_dir),
    ]
    _run(angular)


if __name__ == "__main__":
    main()
