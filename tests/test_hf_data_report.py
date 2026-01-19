from tests import _PATH_DATA
from uaag2.hf_data_report import DATA_FILES, HUGGINGFACE_REPO_ID, build_report


def test_build_report_includes_expected_metadata() -> None:
    """Ensure the Hugging Face data report includes core metadata."""
    report = build_report(_PATH_DATA)
    assert "Hugging Face dataset report" in report
    assert HUGGINGFACE_REPO_ID in report
    for filename in DATA_FILES:
        assert f"`{filename}`" in report
