from transcripto_core.ledger import (
    LEDGER_FILE,
    already_done,
    load_ledger,
    mark_processed,
    save_ledger,
)
from transcripto_core.outputs import OutputLayout, find_collision_stems


def test_missing_ledger_returns_empty(tmp_path):
    assert load_ledger(tmp_path) == {"processed": {}}


def test_save_and_load_roundtrip(tmp_path):
    ledger = {"processed": {"key": {"model": "base"}}}
    save_ledger(tmp_path, ledger)

    assert load_ledger(tmp_path) == ledger
    assert (tmp_path / LEDGER_FILE).exists()
    assert not (tmp_path / f"{LEDGER_FILE}.tmp").exists()


def test_corrupt_ledger_is_ignored(tmp_path):
    (tmp_path / LEDGER_FILE).write_text("{not json", encoding="utf-8")

    assert load_ledger(tmp_path) == {"processed": {}}


def test_already_done_by_resolved_path(tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"x")
    ledger = {"processed": {str(media.resolve()): {}}}

    assert already_done(media, ledger)
    assert not already_done(media, {"processed": {}})


def test_already_done_by_outputs(tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"x")
    layout = OutputLayout()
    stems = find_collision_stems([media], layout)

    assert not already_done(media, {"processed": {}}, layout, stems)
    (tmp_path / "talk.txt").write_text("x")
    (tmp_path / "talk.json").write_text("[]")

    assert already_done(media, {"processed": {}}, layout, stems)


def test_legacy_name_key_is_not_used(tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"x")
    ledger = {"processed": {"talk.mp3": {}}}

    assert not already_done(media, ledger)


def test_mark_processed_records_metadata(tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"x")
    ledger = {"processed": {}}

    mark_processed(ledger, media, "small", 1, 3)
    entry = ledger["processed"][str(media.resolve())]

    assert entry["model"] == "small"
    assert entry["min_speakers"] == 1
    assert entry["max_speakers"] == 3
    assert "timestamp" in entry
