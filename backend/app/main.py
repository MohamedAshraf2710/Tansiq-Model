import logging
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from starlette.requests import Request
from starlette.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database import load_initial_data, get_chat_history, save_chat, upsert_student_profile, get_student_profile
from app.recommender import HybridRecommender
from app.scraper import smart_lookup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Tansiq ML Microservice")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Requested-With"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEY_HEADER = "X-API-Key"

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.url.path in ["/docs", "/openapi.json", "/redoc"]:
        return await call_next(request)
    api_key = request.headers.get(API_KEY_HEADER)
    if not settings.API_SECRET_KEY or api_key == settings.API_SECRET_KEY:
        response = await call_next(request)
        return response
    return JSONResponse(status_code=401, content={"detail": "Invalid API Key"})

# DataOps: Load data pipelines during startup
df, df_geo, df_dist = load_initial_data()
ai_client = genai.Client(api_key=settings.GEMINI_API_KEY)

class InferenceRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    question: Optional[str] = Field("", max_length=2000)
    student_score: Optional[float] = Field(None, ge=0, le=420)
    student_gender: Optional[str] = Field(None, max_length=10)
    student_gov: Optional[str] = Field(None, max_length=50)
    track: Optional[str] = Field(None, max_length=30)
    interests: Optional[List[str]] = Field(default=[], max_length=20)
    priority: Optional[str] = Field("غير محدد", max_length=20)

SYSTEM_INSTRUCTION = """You are "Tansiq Assistant", a professional academic advisor helping Egyptian high school students. 
Tone: Friendly Egyptian Arabic (عامية مصرية ودية). Close the answer with "بالتوفيق يا بطل!" and halt."""

def sanitize_llm_input(text: str, max_len: int = 1000) -> str:
    if not text:
        return ""
    text = text[:max_len]
    injection_patterns = ["ignore previous", "ignore above", "system:", "assistant:", "### instruction"]
    lower = text.lower()
    for pat in injection_patterns:
        if pat in lower:
            text = text.replace(pat, "[filtered]")
    return text.strip()

@app.post("/predict")
@limiter.limit("10/minute")
def predict_answer(payload: InferenceRequest, request: Request):
    try:
        # Check profile variables and upsert/fallback
        if payload.student_score is not None and payload.track is not None:
            upsert_student_profile(
                payload.session_id, payload.student_score, payload.student_gender or "", 
                payload.student_gov or "", payload.track, payload.interests or [], payload.priority or ""
            )
        else:
            profile = get_student_profile(payload.session_id)
            if profile:
                payload.student_score = payload.student_score or profile.get("student_score")
                payload.student_gender = payload.student_gender or profile.get("student_gender")
                payload.student_gov = payload.student_gov or profile.get("student_gov")
                payload.track = payload.track or profile.get("track")
                payload.interests = payload.interests or profile.get("interests", [])
                payload.priority = payload.priority or profile.get("priority")

        # Smart context lookup
        search_result = smart_lookup(payload.question)
        general_context = ""
        faculty_url = None
        if search_result["results"]:
            top_result = search_result["results"][0]
            general_context = top_result.get("text", "") or str(top_result)
            faculty_url = top_result.get("url")
        
        recommender = HybridRecommender(df, df_geo, df_dist)
        
        # Guard clause for missing vital info
        if payload.student_score is None or not payload.track:
            missing_info_response = "أهلاً بيك يا بطل! عشان أقدر أساعدك بشكل دقيق، محتاج أعرف مجموعك كام في الثانوية العامة وإنت قسم إيه (علمي علوم ولا رياضة ولا أدبي)؟"
            save_chat(payload.session_id, payload.question, missing_info_response)
            return {"status": "success", "answer": missing_info_response, "wishes_75": []}

        recommendations = recommender.recommend(
            student_score=payload.student_score, student_gender=payload.student_gender or "",
            student_gov=payload.student_gov or "", track=payload.track,
            interests=payload.interests or [], priority=payload.priority or "غير محدد"
        )

        wishes_75_list = []
        if not recommendations.empty:
            for _, row in recommendations.iterrows():
                wishes_75_list.append({
                    "faculty": str(row.get('Faculty', '')),
                    "governorate": str(row.get('Governorate', '')),
                    "min_score": float(row.get('Score', 0)),
                    "url": str(row.get('URL', '')) if pd.notna(row.get('URL')) else ""
                })
        
        if not payload.interests or payload.priority not in ["تخصص", "محافظة"]:
            fast_response = "أنا سعيد جدا إني بساعدك! مجموعك ما شاء الله ممتاز ومتاح ليك خيارات كتير.\nبس عشان أقدر أرتبلك الكليات بشكل دقيق يفيدك، محتاج أسألك سؤال واحد:\nهل الأهم عندك تدخل الكلية اللي بتحبها حتى لو في محافظة تانية، ولا الأهم تفضل في محافظتك؟"
            save_chat(payload.session_id, payload.question, fast_response)
            return {"status": "success", "answer": fast_response, "wishes_75": wishes_75_list}
            
        recommendation_context = ""
        if not recommendations.empty:
            for _, row in recommendations.head(5).iterrows():
                recommendation_context += f"- كلية {row.get('Faculty')} ({row.get('Governorate')}) تقبل من {row.get('Score')}\n"
        
        priority_text = "تفضل في محافظتك" if payload.priority == "محافظة" else "تدخل التخصص اللي بتحبه"
        chat_history = get_chat_history(payload.session_id)
        
        user_prompt = f"""
        ### Metadata: Score={payload.student_score}, Gov={payload.student_gov}, Track={payload.track}, Priority={priority_text}
        ### Context: {sanitize_llm_input(general_context, 500)}
        ### Top Recommendations: {recommendation_context}
        ### History: {chat_history}
        ### Question: {sanitize_llm_input(payload.question)}
        """

        response = ai_client.models.generate_content(
            model=settings.MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=400
            )
        )
        
        ai_response = response.text
        if faculty_url and not pd.isna(faculty_url) and payload.interests:
             ai_response += "\n\n🔗 للمزيد من التفاصيل، تفضل بزيارة الموقع الرسمي للكلية."

        save_chat(payload.session_id, payload.question, ai_response)
        return {"status": "success", "answer": ai_response, "wishes_75": wishes_75_list}
    except Exception as e:
        logger.error(f"Inference Error: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Internal server error.")