# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import csv
import hashlib
import io
import json
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


VISGYM_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = VISGYM_ROOT / "vendor_wheels" / "gymnasium-1.1.1-py3-none-any.whl.provenance.json"


def test_vendored_visgym_wheel_matches_provenance() -> None:
    provenance = json.loads(PROVENANCE_PATH.read_text())
    wheel_path = PROVENANCE_PATH.parent / provenance["artifact"]

    assert wheel_path.is_file()
    assert hashlib.sha256(wheel_path.read_bytes()).hexdigest() == provenance["sha256"]

    distribution = provenance["distribution"]
    dist_info = f"{distribution['name']}-{distribution['version']}.dist-info"
    with ZipFile(wheel_path) as wheel:
        assert wheel.testzip() is None

        metadata = BytesParser().parsebytes(wheel.read(f"{dist_info}/METADATA"))
        assert metadata["Name"] == distribution["name"]
        assert metadata["Version"] == distribution["version"]

        license_evidence = provenance["license_evidence"]
        declared_metadata = license_evidence["wheel_metadata"]
        assert metadata["License"] == declared_metadata["license"]
        assert declared_metadata["classifier"] in metadata.get_all("Classifier", [])
        assert set(metadata.get_all("License-File", [])) == {
            "LICENSE",
            "LICENSE-GYMNASIUM-v1.1.1",
            "NOTICE",
        }

        wheel_metadata = BytesParser().parsebytes(wheel.read(f"{dist_info}/WHEEL"))
        assert wheel_metadata["Root-Is-Purelib"] == str(distribution["root_is_purelib"]).lower()
        assert wheel_metadata.get_all("Tag") == [distribution["tag"]]

        names = set(wheel.namelist())
        for name in names:
            archive_path = PurePosixPath(name)
            assert not archive_path.is_absolute()
            assert ".." not in archive_path.parts
            assert archive_path.suffix.lower() not in {".dll", ".dylib", ".pyd", ".so"}
        assert "gymnasium/envs/maze_2d/maze_2d.py" in names
        assert "gymnasium/envs/patch_reassembly/patch_reassembly.py" in names

        visgym_license = license_evidence["visgym"]
        assert visgym_license["wheel_license_path"] in names
        assert visgym_license["wheel_notice_path"] in names
        assert b"Apache License" in wheel.read(visgym_license["wheel_license_path"])
        assert b"The VisGym Authors" in wheel.read(visgym_license["wheel_notice_path"])

        gymnasium_license = license_evidence["gymnasium_v1_1_1"]
        assert gymnasium_license["revision"] == "17eff00220d210beda933f78b0e52850022b0690"
        assert gymnasium_license["source_sha256"] == (
            "7dacaa9772e856aee6943b32ef663d3634d91d72ec7bbc74d136943673f91e18"
        )
        local_license_path = (PROVENANCE_PATH.parent / gymnasium_license["local_copy"]).resolve()
        local_license = local_license_path.read_bytes()
        assert hashlib.sha256(local_license).hexdigest() == gymnasium_license["source_sha256"]
        assert wheel.read(gymnasium_license["wheel_license_path"]) == local_license

        added_source_file = provenance["build"]["added_source_file"]
        assert added_source_file["source"] == gymnasium_license["local_copy"]
        assert added_source_file["sha256"] == gymnasium_license["source_sha256"]

        record_rows = {row[0]: row[1:] for row in csv.reader(io.StringIO(wheel.read(f"{dist_info}/RECORD").decode()))}
        encoded_hash = base64.urlsafe_b64encode(hashlib.sha256(local_license).digest()).rstrip(b"=").decode()
        assert record_rows[gymnasium_license["wheel_license_path"]] == [
            f"sha256={encoded_hash}",
            str(len(local_license)),
        ]

        assert license_evidence["review"]["status"] == "maintainer-and-legal-review-required-before-redistribution"
