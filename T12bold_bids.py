
import argparse
import re
import subprocess
from pathlib import Path


def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def require_cmd(name):
    if subprocess.run(["bash", "-lc", f"command -v {name} >/dev/null 2>&1"]).returncode != 0:
        raise RuntimeError(f"Required command not found in PATH: {name}. "
                           f"Open a terminal where FSL is set up (FSLDIR sourced).")


def parse_sub_ses(p):
    s = str(p)
    m_sub = re.search(r"(sub-[^/]+)", s)
    m_ses = re.search(r"(ses-[^/]+)", s)
    if not m_sub:
        raise ValueError(f"Could not parse sub-XX from path: {p}")
    return m_sub.group(1), (m_ses.group(1) if m_ses else None)


def find_n4_t1(bids, sub, ses):
    base = bids / sub

    anat = base / "anat" / "pre"
    if not anat.exists():
        return None

    cands = sorted(anat.glob("*T1w_desc-N4corrected*.nii*"))
    if not cands:
        cands = sorted(anat.glob("*N4*.*nii*"))
    return cands[0] if cands else None


def find_bold_mcf(bids, sub, ses):
    if ses:
        func = bids / sub / ses / "pre"
    else:
        func = bids / sub / "func"
    if not func.exists():
        return None

    cands = []
    for p in func.glob("*.nii*"):
        name = p.name.lower()
        if "boldmcf" in name and "mean" in name:
            cands.append(p)
    cands = sorted(cands)
    return cands[0] if cands else None


def build_out_paths(out_root, sub, ses, t1_path):
    if ses:
        anat_out = out_root / sub / ses / "anat"
        func_out = out_root / sub / ses / "func"
        tag = f"{sub}_{ses}"
    else:
        anat_out = out_root / sub / "anat"
        func_out = out_root / sub / "func"
        tag = f"{sub}"

    anat_out.mkdir(parents=True, exist_ok=True)
    func_out.mkdir(parents=True, exist_ok=True)

    meanbold = func_out / f"{tag}_boldref.nii"
    mat = anat_out / f"{tag}_t12bold.mat"
    warped = anat_out / f"{tag}_t12bold.nii"

    return meanbold, mat, warped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids", required=True, type=Path, help="Path to bids_dir")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Where to write outputs. Default: <BIDS>/derivatives/t1n4_to_boldmcf",
    )
    ap.add_argument("--dof", type=int, default=6, help="FLIRT DOF (6 rigid recommended for T1->BOLD)")
    ap.add_argument("--cost", default="mutualinfo", help="FLIRT cost function (mutualinfo works well cross-modality)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    bids = args.bids.expanduser().resolve()
    if not bids.exists():
        raise SystemExit(f"BIDS dir not found: {bids}")

    out_root = args.out_root.expanduser().resolve() if args.out_root else (bids / "t1_to_bold")

    require_cmd("flirt")
    require_cmd("fslmaths")
    require_cmd("fslval")

    sub_dirs = sorted([p for p in bids.glob("sub-*") if p.is_dir()])

    n_ok = n_fail = n_skip = 0

    for sub_dir in sub_dirs:
        sub = sub_dir.name
        ses_dirs = sorted([p for p in sub_dir.glob("ses-*") if p.is_dir()])

        targets = ses_dirs if ses_dirs else [None]

        for ses_dir in targets:
            ses = ses_dir.name if isinstance(ses_dir, Path) else None

            t1 = find_n4_t1(bids, sub, ses)
            bold_mcf = find_bold_mcf(bids, sub, ses)

            if t1 is None:
                print(f"[SKIP] {sub}{' '+ses if ses else ''}: no N4 T1 found in derivatives/n4biasfield")
                n_skip += 1
                continue
            if bold_mcf is None:
                print(f"[SKIP] {sub}{' '+ses if ses else ''}: no motion-corrected BOLD (*bold* and *mcf* in name) found")
                n_skip += 1
                continue

            meanbold, mat, warped = build_out_paths(out_root, sub, ses, t1)

            if (meanbold.exists() and mat.exists() and warped.exists()) and not args.overwrite:
                print(f"[SKIP] {sub}{' '+ses if ses else ''}: outputs exist")
                n_skip += 1
                continue

            try:
                print(f"\n=== {sub}{' '+ses if ses else ''} ===")
                print(f"  T1(N4):   {t1}")
                print(f"  BOLD mcf: {bold_mcf}")

                dim4 = subprocess.check_output(["bash", "-lc", f"fslval {str(bold_mcf)!s} dim4"]).decode().strip()
                if dim4.isdigit() and int(dim4) > 1:
                    run(["fslmaths", str(bold_mcf), "-Tmean", str(meanbold)])
                else:
                    run(["fslmaths", str(bold_mcf), str(meanbold)])

                run([
                    "flirt",
                    "-in", str(t1),
                    "-ref", str(meanbold),
                    "-omat", str(mat),
                    "-out", str(warped),
                    "-dof", str(args.dof),
                    "-cost", args.cost,
                    "-searchrx", "-90", "90",
                    "-searchry", "-90", "90",
                    "-searchrz", "-90", "90",
                ])

                print(f"[OK] meanbold: {meanbold}")
                print(f"[OK] matrix:   {mat}")
                print(f"[OK] warped:   {warped}")
                n_ok += 1

            except subprocess.CalledProcessError as e:
                print(f"[FAIL] {sub}{' '+ses if ses else ''}: command failed with code {e.returncode}")
                n_fail += 1
            except Exception as e:
                print(f"[FAIL] {sub}{' '+ses if ses else ''}: {e}")
                n_fail += 1

    print(f"\nDone. OK={n_ok}, SKIP={n_skip}, FAIL={n_fail}")
    print(f"Outputs written under: {out_root}")


if __name__ == "__main__":
    main()
