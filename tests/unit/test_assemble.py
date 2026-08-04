import pytest
from tools.ffmpeg.assemble import assemble, AssemblyError


def test_empty_scenes_raises(tmp_path):
    with pytest.raises(AssemblyError, match="No scene videos"):
        assemble(scenes=[], audios=[], output=str(tmp_path / "out.mp4"))


def test_mismatched_scene_audio_counts_raises(tmp_path):
    scene = tmp_path / "scene.mp4"
    audio1 = tmp_path / "a1.wav"
    audio2 = tmp_path / "a2.wav"
    scene.touch()
    audio1.touch()
    audio2.touch()
    with pytest.raises(AssemblyError, match="Mismatched inputs"):
        assemble(
            scenes=[str(scene)],
            audios=[str(audio1), str(audio2)],
            output=str(tmp_path / "out.mp4"),
        )


def test_unknown_quality_raises(tmp_path):
    scene = tmp_path / "scene.mp4"
    audio = tmp_path / "audio.wav"
    scene.touch()
    audio.touch()
    with pytest.raises(AssemblyError, match="Unknown quality"):
        assemble(
            scenes=[str(scene)],
            audios=[str(audio)],
            output=str(tmp_path / "out.mp4"),
            quality="4k",
        )


def test_missing_scene_file_raises(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.touch()
    with pytest.raises(AssemblyError, match="Scene not found"):
        assemble(
            scenes=[str(tmp_path / "does_not_exist.mp4")],
            audios=[str(audio)],
            output=str(tmp_path / "out.mp4"),
        )


def test_missing_audio_file_raises(tmp_path):
    scene = tmp_path / "scene.mp4"
    scene.touch()
    with pytest.raises(AssemblyError, match="Narration audio not found"):
        assemble(
            scenes=[str(scene)],
            audios=[str(tmp_path / "does_not_exist.wav")],
            output=str(tmp_path / "out.mp4"),
        )
