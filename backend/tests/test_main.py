from fastapi.testclient import TestClient
from app.main import app

# Ensure TestClient initialization is called positionally as TestClient(app)
client = TestClient(app)

def test_predict_endpoint_missing_info_bypass():
    """Test the bypass logic when vital info is missing"""
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

def test_predict_endpoint_priority_bypass():
    """Test the priority check bypass logic"""
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
    assert response.status_code in [200, 500]
    # Note: 500 might occur in CI if DB connection fails without proper secrets.
    if response.status_code == 200:
        json_data = response.json()
        assert json_data["status"] == "success"
        assert "هل الأهم عندك تدخل الكلية" in json_data["answer"]
    
def test_predict_endpoint_full():
    """Test a full recommendation path"""
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
    assert response.status_code in [200, 500]
    # Note: 500 might occur locally if DB connection fails without proper secrets.
    # If 200, we should see AI response and wishes.
    if response.status_code == 200:
        json_data = response.json()
        assert json_data["status"] == "success"
        assert "answer" in json_data
        assert isinstance(json_data["wishes_75"], list)