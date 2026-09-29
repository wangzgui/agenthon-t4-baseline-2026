"""Make the two-file CodaBench ZIP for S1.2 using the official packer.

Usage: python prepare_submission_s12.py sha256:<published image digest>
Read the Agenthon Team Key from stdin; never place it on the command line.
"""

import json
import hashlib
import pathlib
import re
import sys
import zipfile

import qfbench2_common
from qfbench2_common.team_claim import pack_submission, validate_team_key

TEAM_NUMBER = 574
REPOSITORY = "wangzgui/agenthon-t4-baseline-2026"


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"sha256:[0-9a-f]{64}", sys.argv[1]):
        raise SystemExit("expected immutable sha256 image digest")
    root = pathlib.Path(__file__).resolve().parent
    fixture = pathlib.Path(qfbench2_common.__file__).parent / "contracts/fixtures/c5/analysis_dev.json"
    descriptor = json.loads(fixture.read_text(encoding="utf-8"))
    descriptor.pop("team_id", None)
    descriptor.pop("descriptor_digest", None)
    descriptor["schema_version"] = "1.1.0"
    descriptor["image"] = {"registry": "ghcr.io", "repository": REPOSITORY, "digest": sys.argv[1]}
    descriptor["image_access"] = "public"
    local_revision = hashlib.sha256((root / "s12_numerics.py").read_bytes()).hexdigest()
    descriptor["models"] = [{
        "name": "nvidia/nemotron-3-super-120b-a12b",
        "version": "rl-030326-fp8",
        "revision": "rl-030326-fp8",
        "training_cutoff": "unpublished",
        "access": "api",
    }, {
        "name": "s12-cutoff-corpus-rolling-ridge",
        "version": "1.0.0",
        "revision": "sha256:" + local_revision,
        "training_cutoff": "per-unit task.cutoff_date; supplied frozen corpus only",
        "access": "local",
    }]
    descriptor["license"] = "MIT"
    key = validate_team_key(sys.stdin.read().rstrip("\r\n"))
    archive = root / "submission-s1.2.zip"
    pack_submission(descriptor, TEAM_NUMBER, key, archive)
    with zipfile.ZipFile(archive) as zf:
        assert set(zf.namelist()) == {"submission.json", "team-claim.json"}
        (root / "submission-s1.2.json").write_bytes(zf.read("submission.json"))
    print(f"Packed {archive}; Team Key excluded")


if __name__ == "__main__":
    main()
