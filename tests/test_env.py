import os

from transcripto_core.env import Settings, env_bool, env_int, prepare_environment


def test_env_bool(monkeypatch):
    monkeypatch.setenv("FLAG", "yes")
    assert env_bool("FLAG") is True
    monkeypatch.setenv("FLAG", "0")
    assert env_bool("FLAG") is False
    monkeypatch.delenv("FLAG")
    assert env_bool("FLAG", True) is True


def test_env_int(monkeypatch):
    monkeypatch.setenv("COUNT", "3")
    assert env_int("COUNT", 1) == 3
    monkeypatch.setenv("COUNT", "not-a-number")
    assert env_int("COUNT", 1) == 1


def test_settings_default_min_speakers_is_one(monkeypatch):
    for name in ("MODEL_SIZE", "DEVICE", "COMPUTE_TYPE", "FORCE_CPU", "ENABLE_TF32",
                 "MIN_SPEAKERS", "MAX_SPEAKERS", "HF_TOKEN", "LANGUAGE"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.min_speakers == 1
    assert settings.max_speakers >= settings.min_speakers
    assert settings.model_size == "base"
    assert settings.device == "auto"
    assert settings.hf_token is None


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MODEL_SIZE", "small")
    monkeypatch.setenv("DEVICE", "CPU")
    monkeypatch.setenv("MIN_SPEAKERS", "2")
    monkeypatch.setenv("MAX_SPEAKERS", "4")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("FORCE_CPU", "1")

    settings = Settings.from_env()

    assert settings.model_size == "small"
    assert settings.device == "cpu"
    assert settings.min_speakers == 2
    assert settings.max_speakers == 4
    assert settings.hf_token == "token"
    assert settings.force_cpu is True


def test_prepare_environment_sets_caches(monkeypatch, tmp_path):
    for name in ("TRANSFORMERS_NO_TORCHVISION", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME"):
        monkeypatch.delenv(name, raising=False)
    app_folder = tmp_path / "app"
    app_folder.mkdir()

    prepare_environment(app_folder)

    assert os.environ["TRANSFORMERS_NO_TORCHVISION"] == "1"
    assert os.environ["HF_HOME"] == str((app_folder / "_hf_cache").resolve())
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str((app_folder / "_hf_cache" / "hub").resolve())
    assert os.environ["TORCH_HOME"] == str((app_folder / "_torch_cache").resolve())
    assert (app_folder / "_hf_cache" / "hub").is_dir()
    assert (app_folder / "_torch_cache").is_dir()
