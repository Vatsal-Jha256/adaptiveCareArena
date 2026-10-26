import json

from runtime.manifest import write_results


def test_write_results_records_exact_runner_resources_when_supplied(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({"experiment_id": "resource_test"}), encoding="utf-8"
    )
    write_results(tmp_path, {"value": 1.0}, wall_seconds=12.5)
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["resources"]["wall_seconds"] == 12.5
    assert manifest["resources"]["peak_rss_kb"] > 0
    assert manifest["resources"]["measurement_scope"] == "current runner process"


def test_write_results_does_not_invent_timing_for_legacy_callers(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({"experiment_id": "resource_test"}), encoding="utf-8"
    )
    write_results(tmp_path, {"value": 1.0})
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert "resources" not in manifest
