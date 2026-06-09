from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TEST_ROOT.parent
FIXTURE_ROOT = TEST_ROOT / "fixtures"
PERFLIB_FIXTURE_ROOT = FIXTURE_ROOT / "perflib"
PORT_FIXTURE_ROOT = FIXTURE_ROOT / "port"
PORT_TRACKER_FIXTURE_ROOT = PORT_FIXTURE_ROOT / "tracker"
