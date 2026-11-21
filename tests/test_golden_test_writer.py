import csv
from pathlib import Path

from golden_test.run_golden_test import _write_results


def test_write_results_uses_fallback_when_default_is_locked(tmp_path, monkeypatch):
    results_path = tmp_path / "golden_test_results.csv"
    original_open = Path.open

    def open_with_locked_results(path, *args, **kwargs):
        if path == results_path:
            raise PermissionError("file is locked")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_locked_results)

    actual_path = _write_results(
        [{"query_id": "Q001", "status": "answered"}],
        results_path,
    )

    assert actual_path != results_path
    assert actual_path.exists()
    with original_open(actual_path, encoding="utf-8", newline="") as result_file:
        assert list(csv.DictReader(result_file)) == [
            {"query_id": "Q001", "status": "answered"}
        ]
