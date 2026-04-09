
import argparse
import os
import shlex
import subprocess
from pathlib import Path


def find_t1w_files(bids_dir):

    pattern = "sub-*/anat/T1w.nii"

    files: List[Path] = []
    files.extend(sorted(bids_dir.glob(pattern)))
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def parse_sub(t1_path):
    parts = t1_path.parts
    sub = next(p for p in parts if p.startswith("sub-"))

    return sub


def run_fastsurfer(fastsurfer_cmd, t1_path, sd_dir, sid, threads, py_bin="python3", extra_args=None, dry_run=False):
    sd_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        fastsurfer_cmd,
        "--t1",
        str(t1_path),
        "--sd",
        str(sd_dir),
        "--sid",
        sid,
        "--threads",
        str(threads),
        "--py",
        py_bin,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print("\n>>>", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return

    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids",
                    type=Path,
                    default=Path("/Users/mac/PycharmProjects/pythonMPhysproject/bids_dir"),
                    help="Path to BIDS root (bids_dir)")
    ap.add_argument(
        "--out-mode",
        choices=["derivatives", "session"],
        default="derivatives",
        help="Where to write outputs: derivatives/fastsurfer/... (default) or inside each ses folder",
    )
    ap.add_argument(
        "--deriv-name",
        default="fastsurfer",
        help="Name under derivatives/ (only used if --out-mode derivatives)",
    )
    ap.add_argument(
        "--sid",
        default="fastsurfer",
        help="FastSurfer subject id folder name created inside --sd (keep constant to avoid nesting repeats)",
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--fastsurfer-cmd",
        default="/Users/mac/PycharmProjects/pythonMPhysproject/FastSurfer/run_fastsurfer.sh",
        help="FastSurfer entrypoint (e.g., FastSurferCNN or path/to/run_fastsurfer.sh)",
    )
    ap.add_argument("--py", default="python3", help="Python executable to pass to FastSurfer (--py)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    ap.add_argument(
        "--extra",
        default="--fs_license /Users/mac/PycharmProjects/pythonMPhysproject/FastSurfer/LICENSE",
        help='Extra args passed to FastSurfer as a single string, e.g. \'--device cuda --batch_size 4\'',
    )
    args = ap.parse_args()

    bids_dir: Path = args.bids.resolve()
    if not bids_dir.exists():
        raise FileNotFoundError(f"BIDS dir not found: {bids_dir}")

    t1_files = find_t1w_files(bids_dir)
    if not t1_files:
        raise RuntimeError(f"No T1w files found under {bids_dir} (sub-*/anat/T1w.nii)")

    extra_args = shlex.split(args.extra) if args.extra.strip() else None

    print(f"Found {len(t1_files)} T1w files.")

    for t1 in t1_files:
        sub = parse_sub(t1)

        if args.out_mode == "derivatives":
            sd_dir = bids_dir / "derivatives" / args.deriv_name / sub
        else:
            sd_dir = bids_dir / sub

        try:
            run_fastsurfer(
                fastsurfer_cmd=args.fastsurfer_cmd,
                t1_path=t1,
                sd_dir=sd_dir,
                sid=args.sid,
                threads=args.threads,
                py_bin=args.py,
                extra_args=extra_args,
                dry_run=args.dry_run,
            )
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] FastSurfer failed for {sub} : {t1}")
            print(f"Command returned code {e.returncode}. Continuing to next.\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
