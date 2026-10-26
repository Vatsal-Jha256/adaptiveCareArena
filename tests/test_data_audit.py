from __future__ import annotations

import adaptivecarearena.data_audit as data_audit


def test_data_audit_emits_aggregates_without_identifiers(monkeypatch, tmp_path):
    trajectories = {
        10: [0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -2.0],
        20: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    monkeypatch.setattr(
        data_audit,
        "load_panel",
        lambda cfg: ([], {}, trajectories, {}),
    )
    monkeypatch.setattr(
        data_audit,
        "data_fingerprint",
        lambda root: {
            "exists": True,
            "n_files": 2,
            "combined_sha256": "abc123",
            "files": [{"path": "sensitive.csv"}],
        },
    )
    report = data_audit.audit_config(
        {
            "data": {"source": "ces", "local_root": str(tmp_path)},
            "outcome": {"threshold": -1.0, "horizon_days": 1},
        }
    )
    assert report["participants"] == 2
    assert report["recorded_reports"] == 14
    assert report["eligible_proxy_episodes"] == 1
    assert report["data_combined_sha256"] == "abc123"
    serialized = str(report)
    assert "sensitive.csv" not in serialized
    assert "10" not in serialized and "20" not in serialized
