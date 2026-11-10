from pathlib import Path

from transcripto_core.outputs import (
    OutputLayout,
    collect_folder_media,
    find_collision_stems,
    format_timestamp,
    map_speakers_to_letters,
    merge_speaker_turns,
    output_paths,
    outputs_exist,
    render_merged,
    render_srt,
    render_txt,
    render_vtt,
    write_outputs,
)


def make_media(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"media")
    return path


def test_single_file_keeps_stem(tmp_path):
    media = make_media(tmp_path / "solo.mp3")
    layout = OutputLayout()
    stems = find_collision_stems([media], layout)

    assert output_paths(media, layout, stems) == (
        tmp_path / "solo.txt",
        tmp_path / "solo.json",
    )


def test_same_stem_in_same_folder_keeps_extension(tmp_path):
    first = make_media(tmp_path / "talk.mp4")
    second = make_media(tmp_path / "talk.mp3")
    layout = OutputLayout()
    stems = find_collision_stems([first, second], layout)

    assert output_paths(first, layout, stems)[0] == tmp_path / "talk.mp4.txt"
    assert output_paths(second, layout, stems)[0] == tmp_path / "talk.mp3.txt"


def test_same_stem_in_different_folders_does_not_collide(tmp_path):
    first = make_media(tmp_path / "a" / "talk.mp4")
    second = make_media(tmp_path / "b" / "talk.mp3")
    layout = OutputLayout()
    stems = find_collision_stems([first, second], layout)

    assert output_paths(first, layout, stems)[0] == tmp_path / "a" / "talk.txt"
    assert output_paths(second, layout, stems)[0] == tmp_path / "b" / "talk.txt"


def test_output_dir_mirrors_input_tree(tmp_path):
    root = tmp_path / "library"
    media = make_media(root / "shows" / "episode.mp3")
    layout = OutputLayout(output_dir=tmp_path / "out", roots=(root,))

    txt, json_path = output_paths(media, layout, set())

    assert txt == tmp_path / "out" / "shows" / "episode.txt"
    assert json_path == tmp_path / "out" / "shows" / "episode.json"


def test_collect_folder_media_adds_siblings(tmp_path):
    selected = make_media(tmp_path / "talk.mp4")
    sibling = make_media(tmp_path / "talk.mp3")
    other = make_media(tmp_path / "other.wav")

    assert collect_folder_media([selected]) == [other, sibling, selected]


def test_outputs_exist_uses_collision_layout(tmp_path):
    media = make_media(tmp_path / "talk.mp4")
    layout = OutputLayout()
    stems = {(tmp_path, "talk")}

    assert not outputs_exist(media, layout, stems)
    (tmp_path / "talk.mp4.txt").write_text("x")
    (tmp_path / "talk.mp4.json").write_text("[]")

    assert outputs_exist(media, layout, stems)


SEGMENTS = [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.25, "text": "hello"},
    {"speaker": "SPEAKER_00", "start": 1.25, "end": 2.0, "text": "again"},
    {"speaker": "SPEAKER_01", "start": 2.0, "end": 65.5, "text": "hi"},
]


def test_render_txt_and_speaker_mapping():
    segments = map_speakers_to_letters([dict(seg) for seg in SEGMENTS])
    text = render_txt(segments)

    assert "[0.0s - 1.2s] Speaker A: hello" in text
    assert "[2.0s - 65.5s] Speaker B: hi" in text


def test_render_srt_and_vtt():
    segments = map_speakers_to_letters([dict(seg) for seg in SEGMENTS])

    srt = render_srt(segments)
    vtt = render_vtt(segments)

    assert "1\n00:00:00,000 --> 00:00:01,250\nSpeaker A: hello" in srt
    assert "00:01:05,500" in srt
    assert vtt.startswith("WEBVTT")
    assert "00:01:05.500" in vtt


def test_merged_transcript_groups_turns():
    segments = map_speakers_to_letters([dict(seg) for seg in SEGMENTS])

    assert merge_speaker_turns(segments) == [
        ("Speaker A", "hello again"),
        ("Speaker B", "hi"),
    ]
    assert render_merged(segments) == "Speaker A: hello again\n\nSpeaker B: hi"


def test_write_outputs_all_formats(tmp_path):
    media = make_media(tmp_path / "talk.mp4")
    layout = OutputLayout()
    stems = find_collision_stems([media], layout)
    segments = map_speakers_to_letters([dict(seg) for seg in SEGMENTS])

    paths = write_outputs(
        media,
        segments,
        layout,
        stems,
        formats=["txt", "json", "srt", "vtt"],
        merged=True,
    )

    assert set(paths) == {"txt", "json", "srt", "vtt", "merged"}
    for path in paths.values():
        assert path.exists()
    assert paths["srt"].suffix == ".srt"
    assert paths["merged"].name == "talk.merged.txt"


def test_format_timestamp_rounds_milliseconds():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(65.499) == "00:01:05,499"
    assert format_timestamp(3661.5) == "01:01:01,500"
