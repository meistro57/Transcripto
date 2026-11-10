from transcripto_core.summary import SummaryWriter

SEGMENTS = [
    {"speaker_label": "Speaker A", "start": 0.0, "end": 1.0, "text": "hello"},
    {"speaker_label": "Speaker B", "start": 1.0, "end": 2.0, "text": "hi"},
]


def test_no_file_until_first_add(tmp_path):
    path = tmp_path / "summary.txt"
    writer = SummaryWriter(path)

    assert not path.exists()
    writer.close()
    assert not path.exists()


def test_add_streams_content(tmp_path):
    path = tmp_path / "summary.txt"
    with SummaryWriter(path) as writer:
        writer.add("one.mp3", SEGMENTS)
        assert writer.written is True
        first = path.read_text(encoding="utf-8")
        writer.add("two.mp3", SEGMENTS)

    content = path.read_text(encoding="utf-8")

    assert first.startswith("=== one.mp3 ===")
    assert content.count("=== ") == 2
    assert "hello" in content and "hi" in content
