"""Seal the Agenthon T4 Development descriptor with the official toolkit.

Read the Team Key from stdin. Never pass it on the command line or print it.
"""

import json
import pathlib
import sys
import zipfile

import qfbench2_common
from qfbench2_common.team_claim import pack_submission, validate_team_key


IMAGE_DIGEST = "sha256:a6d40dd13ef533277d3c1a4335fe383dd684ef22ccf19051d7fb77981400ea2d"
TEAM_NUMBER = 574


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    fixture = (
        pathlib.Path(qfbench2_common.__file__).parent
        / "contracts/fixtures/c5/analysis_dev.json"
    )
    descriptor = json.loads(fixture.read_text(encoding="utf-8"))
    descriptor.pop("team_id")
    descriptor.pop("descriptor_digest")
    descriptor["schema_version"] = "1.1.0"
    descriptor["image"] = {
        "registry": "ghcr.io",
        "repository": "wangzgui/agenthon-t4-baseline-2026",
        "digest": IMAGE_DIGEST,
    }
    descriptor["image_access"] = "public"
    descriptor["models"] = [
        {
            "name": "nvidia/nemotron-3-super-120b-a12b",
            "version": "rl-030326-fp8",
            "revision": "rl-030326-fp8",
            "training_cutoff": "unpublished",
            "access": "api",
        }
    ]
    descriptor["license"] = "MIT"

    key = validate_team_key(sys.stdin.read().rstrip("\r\n"))
    archive = root / "submission.zip"
    team_id = pack_submission(descriptor, TEAM_NUMBER, key, archive)
    with zipfile.ZipFile(archive) as zf:
        assert set(zf.namelist()) == {"submission.json", "team-claim.json"}
        (root / "submission.json").write_bytes(zf.read("submission.json"))
    print(f"Packed {archive.name} for {team_id}; Team Key excluded")


if __name__ == "__main__":
    main()
