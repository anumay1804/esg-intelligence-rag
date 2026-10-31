from pathlib import Path

from config.settings import settings


def test_settings_have_local_paths():
    raw_dir = Path(settings.raw_dir)
    processed_dir = Path(settings.processed_dir)

    assert raw_dir.parts[-2:] == ("data", "raw")
    assert processed_dir.parts[-2:] == ("data", "processed")