from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app=app)

def test_predict_endpoint_bypass_logic():
  
    payload = {
        "session_id": "test_session_123",
        "question": "عايز مجموع كليات حاسبات",
        "student_score": 380.0,
        "student_gender": "ذكر",
        "student_gov": "المنوفية",
        "track": "علمي رياضة",
        "interests": [],
        "priority": "غير محدد"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert "الأهم عندك تدخل الكلية اللي بتحبها" in json_data["answer"]