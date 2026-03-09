"""Sample tests to verify CI pipeline works."""
import sys
from pathlib import Path

import pytest

# Add parent directory to sys.path so we can import app.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import AnalysisResponse


class TestAnalysisResponse:
    """Tests for the AnalysisResponse model."""

    def test_analysis_response_creation(self):
        """Test that AnalysisResponse can be created with valid data."""
        response = AnalysisResponse(
            chat_reply="Here is the analysis",
            plan_steps=["Step 1", "Step 2"],
            code="print('hello')"
        )
        assert response.chat_reply == "Here is the analysis"
        assert len(response.plan_steps) == 2
        assert response.code == "print('hello')"

    def test_analysis_response_defaults(self):
        """Test that AnalysisResponse uses default values."""
        response = AnalysisResponse(chat_reply="Just a reply")
        assert response.chat_reply == "Just a reply"
        assert response.plan_steps == []
        assert response.code == ""

    def test_analysis_response_empty_plan(self):
        """Test that conversational queries can have empty plan_steps."""
        response = AnalysisResponse(
            chat_reply="Conversational answer",
            plan_steps=[],
            code=""
        )
        assert response.plan_steps == []
        assert response.code == ""


def test_module_imports():
    """Test that app module imports successfully."""
    import app
    assert hasattr(app, "AnalysisState")
    assert hasattr(app, "AnalysisResponse")
    assert hasattr(app, "clear_artifacts")
    assert hasattr(app, "save_template")
