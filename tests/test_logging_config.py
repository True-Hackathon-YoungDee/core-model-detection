import os
import sys

from fall_detection.logging_config import quiet_native_logs


def test_quiet_native_logs_sets_glog_and_tf_vars(monkeypatch):
    monkeypatch.delenv("GLOG_minloglevel", raising=False)
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)

    quiet_native_logs()

    assert os.environ["GLOG_minloglevel"] == "2"
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "3"


def test_quiet_native_logs_forces_xcb_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)

    quiet_native_logs()

    assert os.environ["QT_QPA_PLATFORM"] == "xcb"


def test_quiet_native_logs_does_not_override_explicit_qt_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")

    quiet_native_logs()

    assert os.environ["QT_QPA_PLATFORM"] == "wayland"


def test_quiet_native_logs_rewrites_wayland_session_type_to_x11(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")

    quiet_native_logs()

    assert os.environ["XDG_SESSION_TYPE"] == "x11"


def test_quiet_native_logs_leaves_non_wayland_session_type_alone(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")

    quiet_native_logs()

    assert os.environ["XDG_SESSION_TYPE"] == "x11"


def test_quiet_native_logs_sets_qt_logging_rules(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("QT_LOGGING_RULES", raising=False)

    quiet_native_logs()

    assert os.environ["QT_LOGGING_RULES"] == "*.warning=false"


def test_quiet_native_logs_does_not_override_explicit_qt_logging_rules(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("QT_LOGGING_RULES", "*.debug=true")

    quiet_native_logs()

    assert os.environ["QT_LOGGING_RULES"] == "*.debug=true"


def test_quiet_native_logs_sets_fontdir_when_a_system_font_dir_exists(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("QT_QPA_FONTDIR", raising=False)
    fake_fonts = tmp_path / "fonts"
    fake_fonts.mkdir()
    monkeypatch.setattr(
        "fall_detection.logging_config._LINUX_FONT_DIR_CANDIDATES",
        (str(fake_fonts),),
    )

    quiet_native_logs()

    assert os.environ["QT_QPA_FONTDIR"] == str(fake_fonts)


def test_quiet_native_logs_skips_fontdir_when_none_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("QT_QPA_FONTDIR", raising=False)
    monkeypatch.setattr(
        "fall_detection.logging_config._LINUX_FONT_DIR_CANDIDATES",
        (str(tmp_path / "does-not-exist"),),
    )

    quiet_native_logs()

    assert "QT_QPA_FONTDIR" not in os.environ


def test_quiet_native_logs_leaves_qt_vars_alone_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("QT_QPA_FONTDIR", raising=False)

    quiet_native_logs()

    assert "QT_QPA_PLATFORM" not in os.environ
    assert "QT_QPA_FONTDIR" not in os.environ


def test_quiet_native_logs_leaves_qt_vars_alone_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("QT_QPA_FONTDIR", raising=False)

    quiet_native_logs()

    assert "QT_QPA_PLATFORM" not in os.environ
    assert "QT_QPA_FONTDIR" not in os.environ
