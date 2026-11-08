"""The hang recorder writes every thread's stack once per freeze, and nothing while alive."""
import time

from PySide6.QtCore import QCoreApplication


def test_watchdog_reports_a_blocked_main_thread_once(qapp, tmp_path):
    from rsstatchem.hang_watchdog import HangWatchdog

    dog = HangWatchdog(tmp_path / "logs", limit=0.4, interval_ms=50)
    try:
        for _ in range(20):                       # responsive: no report
            QCoreApplication.processEvents()
            time.sleep(0.03)
        assert dog.files == []
        time.sleep(1.2)                           # the main thread "hangs"
        assert len(dog.files) == 1                # one report for the whole episode
        text = dog.files[0].read_text(encoding="utf-8")
        assert "did not respond" in text and "test_watchdog_reports" in text
        for _ in range(10):                       # alive again, then a second freeze
            QCoreApplication.processEvents()
            time.sleep(0.06)
        time.sleep(1.2)
        assert len(dog.files) == 2 and dog.files[1] != dog.files[0]
    finally:
        dog.stop()
