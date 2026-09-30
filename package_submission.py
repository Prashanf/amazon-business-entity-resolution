#!/usr/bin/env python3
"""
Official Submission Packager & Validator for Amazon ML Challenge 2026.
Builds the required <team_name>_submission.zip matching the exact specification:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (scored on leaderboard)
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all modular source code
│       │   ├── __init__.py
│       │   ├── preprocessing.py
│       │   ├── gpu_blocking.py
│       │   ├── feature_engineering.py
│       │   ├── train.py
│       │   └── inference.py
│       ├── README.md               # reproduction guide
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled methodology writeup
"""

import os
import sys
import zipfile
import subprocess
import argparse

def validate_submission_files(matching_tsv, candidate_tsv, test_dir):
    """Runs the official validator script on the output files."""
    validator_path = "student_resource/utils/validate_submission.py"
    if not os.path.exists(validator_path):
        validator_path = "utils/validate_submission.py"
    
    if not os.path.exists(validator_path):
        print(f"⚠️ Warning: Validator script {validator_path} not found. Skipping validation.")
        return True

    print("\n" + "=" * 75)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 75)
    cmd = [
        sys.executable, validator_path,
        "--matching", matching_tsv,
        "--candidate", candidate_tsv,
        "--test-dir", test_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode == 0:
        print("🎉 VALIDATION PASSED (Exit Code 0)! Output files strictly conform to competition rules.")
        return True
    else:
        print("❌ VALIDATION FAILED:")
        print(res.stderr)
        return False

def package_submission(
    team_name="DataResolvers",
    output_dir="output",
    code_dir="code/business_entity_resolution",
    doc_path="Documentation_template.md",
    zip_path=None,
    test_dir="student_resource/dataset/test",
    run_validation=True
):
    """Constructs the official submission zip archive."""
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")

    if not os.path.exists(matching_tsv):
        raise FileNotFoundError(f"Missing required file: {matching_tsv}")
    if not os.path.exists(candidate_tsv):
        raise FileNotFoundError(f"Missing required file: {candidate_tsv}")
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Missing required documentation: {doc_path}")
    if not os.path.exists(code_dir):
        raise FileNotFoundError(f"Missing required code directory: {code_dir}")

    # Validate before zipping
    if run_validation and os.path.exists(test_dir):
        is_valid = validate_submission_files(matching_tsv, candidate_tsv, test_dir)
        if not is_valid:
            print("\n⚠️ WARNING: Submission validator reported issues. Continuing zip generation...")

    if zip_path is None:
        zip_path = f"{team_name}_submission.zip"

    print("\n" + "=" * 75)
    print(f"ASSEMBLING OFFICIAL SUBMISSION ARCHIVE: {zip_path}")
    print("=" * 75)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        zf.write(matching_tsv, arcname=f"output/matching_results.tsv")
        zf.write(candidate_tsv, arcname=f"output/candidate_pairs.tsv")
        print(f"  + Added: output/matching_results.tsv ({os.path.getsize(matching_tsv):,} bytes)")
        print(f"  + Added: output/candidate_pairs.tsv ({os.path.getsize(candidate_tsv):,} bytes)")

        # 2. Code artefacts
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                if file.endswith((".py", ".txt", ".md")) and not file.startswith("."):
                    abs_file = os.path.join(root, file)
                    rel_arc = os.path.relpath(abs_file, os.path.dirname(code_dir))
                    zf.write(abs_file, arcname=f"code/{rel_arc}")
                    print(f"  + Added: code/{rel_arc}")

        # 3. Documentation template
        zf.write(doc_path, arcname="Documentation_template.md")
        print(f"  + Added: Documentation_template.md ({os.path.getsize(doc_path):,} bytes)")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print("\n" + "=" * 75)
    print(f"🎉 SUBMISSION ZIP READY: {zip_path} ({zip_size_mb:.2f} MB)")
    print("=" * 75)
    print("Archive Contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"  • {info.filename:<50} ({info.file_size:,} bytes)")
    return zip_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Amazon ML Challenge 2026 Submission Zip")
    parser.add_argument("--team-name", default="DataResolvers", help="Your team name")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Test directory path")
    parser.add_argument("--no-validate", action="store_true", help="Skip validator script check")
    args = parser.parse_args()

    package_submission(
        team_name=args.team_name,
        test_dir=args.test_dir,
        run_validation=not args.no_validate
    )
