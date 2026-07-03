from setuptools.build_meta import *

from setuptools.build_meta import build_wheel as _build_wheel


def build_wheel(wheel_directory, config_settings=None,metadata_directory=None, **_kwargs):
    print("=" * 80)
    print("Flash attention is required but not installed through setup.py. Install it manually through https://github.com/mjun0812/flash-attention-prebuild-wheels")
    print("=" * 80)
    
    return _build_wheel(
        wheel_directory,
        config_settings,
        metadata_directory,
    )