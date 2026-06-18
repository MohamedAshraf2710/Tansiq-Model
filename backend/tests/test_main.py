from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@patch('app.main.upsert_student_profile')
@patch('app.main.get_student_profile')
@patch('app.main.save_chat')
def test_predict_endpoint_missing_info_bypass(mock_save_chat, mock_get_profile, mock_upsert):
    """Test the bypass logic when vital info is missing"""
    mock_get_profile.return_value = {}
    payload = {
        "session_id": "test_session_bypass_123",
        "question": "انا عايز ادخل هندسة",
        "student_score": None,
        "track": None
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert "محتاج أعرف مجموعك كام" in json_data["answer"]
    assert len(json_data["wishes_75"]) == 0
    mock_save_chat.assert_called_once()

@patch('app.main.upsert_student_profile')
@patch('app.main.get_student_profile')
@patch('app.main.save_chat')
@patch('app.main.get_chat_history')
@patch('app.main.smart_lookup')
@patch('app.main.ai_client.chat.completions.create', new_callable=AsyncMock)
def test_predict_endpoint_priority_bypass(mock_gen_content, mock_lookup, mock_get_history, mock_save_chat, mock_get_profile, mock_upsert):
    """Test the full path correctly handles a general query and returns mocked AI response."""
    mock_lookup.return_value = {"results": [], "source": "none"}
    mock_get_history.return_value = ""
    
    mock_choice = MagicMock()
    mock_choice.message.content = "Mocked AI Response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_gen_content.return_value = mock_response

    payload = {
        "session_id": "test_session_priority_123",
        "question": "عايز كليات حاسبات",
        "student_score": 380.0,
        "student_gender": "ذكر",
        "student_gov": "المنوفية",
        "track": "علمي رياضة",
        "interests": ["حاسبات", "برمجة"],
        "priority": "غير محدد"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert json_data["answer"] == "Mocked AI Response"
    mock_upsert.assert_called_once()
    mock_gen_content.assert_called_once()
    mock_save_chat.assert_called_once()
    
@patch('app.main.upsert_student_profile')
@patch('app.main.save_chat')
@patch('app.main.get_chat_history')
@patch('app.main.smart_lookup')
@patch('app.main.ai_client.chat.completions.create', new_callable=AsyncMock)
def test_predict_endpoint_full(mock_gen_content, mock_lookup, mock_get_history, mock_save_chat, mock_upsert):
    """Test a full recommendation path with valid payload"""
    mock_lookup.return_value = {"results": [], "source": "none"}
    mock_get_history.return_value = ""
    
    mock_choice = MagicMock()
    mock_choice.message.content = "AI Full Path Response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_gen_content.return_value = mock_response

    payload = {
        "session_id": "test_session_full_123",
        "question": "ايه احسن كليات لعلمي علوم؟",
        "student_score": 395.0,
        "student_gender": "انثى",
        "student_gov": "القاهرة",
        "track": "علمي علوم",
        "interests": ["طب", "علاج طبيعي"],
        "priority": "تخصص"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert "answer" in json_data
    assert json_data["answer"] == "AI Full Path Response"
    assert isinstance(json_data["wishes_75"], list)
    mock_upsert.assert_called_once()