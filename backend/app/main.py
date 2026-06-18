import asyncio
import logging
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import AsyncGroq
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
ai_client = AsyncGroq(api_key=settings.GROQ_API_KEY)

class InferenceRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    question: Optional[str] = Field("", max_length=2000)
    student_score: Optional[float] = Field(None, ge=0, le=420)
    student_gender: Optional[str] = Field(None, max_length=10)
    student_gov: Optional[str] = Field(None, max_length=50)
    track: Optional[str] = Field(None, max_length=30)
    interests: Optional[List[str]] = Field(default=[], max_length=20)
    priority: Optional[str] = Field("غير محدد", max_length=20, description="Deprecated")
    refinements: Optional[List[dict]] = Field(default=None)

SYSTEM_INSTRUCTION = """You are "Tansiq Assistant", a professional academic advisor helping Egyptian high school students. 
Strictly cite and use only the faculties provided in the `wishes_75` metadata.
Do not invent or hallucinate any faculty information.
Maintain a highly professional counseling tone in Friendly Egyptian Arabic (عامية مصرية ودية).
Close the answer with "بالتوفيق يا بطل!" and halt."""

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
async def predict_answer(request: Request, payload: InferenceRequest): # pylint: disable=unused-argument
    try:
        # Check profile variables and upsert/fallback
        if payload.student_score is not None and payload.track is not None:
            await asyncio.to_thread(
                upsert_student_profile,
                payload.session_id, payload.student_score, payload.student_gender or "", 
                payload.student_gov or "", payload.track, payload.interests or [], payload.priority or ""
            )
        else:
            profile = await asyncio.to_thread(get_student_profile, payload.session_id)
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
            await asyncio.to_thread(save_chat, payload.session_id, payload.question, missing_info_response)
            return {"status": "success", "answer": missing_info_response, "wishes_75": []}

        recommendations = recommender.recommend(
            student_score=payload.student_score, student_gender=payload.student_gender or "",
            student_gov=payload.student_gov or "", track=payload.track,
            interests=payload.interests or [], priority=payload.priority or "غير محدد",
            refinements=payload.refinements
        )

        wishes_75_list = []
        if recommendations:
            for row in recommendations:
                wishes_75_list.append({
                    "faculty": str(row.get('Faculty', '')),
                    "governorate": str(row.get('Governorate', '')),
                    "min_score": float(row.get('Score', 0)),
                    "url": str(row.get('URL', '')) if pd.notna(row.get('URL')) else ""
                })
        

        recommendation_context = ""
        if recommendations:
            for row in recommendations[:5]:
                rec_text = f"- كلية {row.get('Faculty')} ({row.get('Governorate')}) تقبل من {row.get('Score')}"
                if pd.notna(row.get('رؤية الكلية')) and row.get('رؤية الكلية'):
                    rec_text += f" | الرؤية: {str(row.get('رؤية الكلية'))[:100]}..."
                if pd.notna(row.get('general_departments')) and row.get('general_departments'):
                    rec_text += f" | الأقسام: {row.get('general_departments')}"
                if pd.notna(row.get('special_sections')) and row.get('special_sections'):
                    rec_text += f" | خاص: {row.get('special_sections')}"
                if pd.notna(row.get('URL')) and row.get('URL'):
                    rec_text += f" | الموقع: {row.get('URL')}"
                recommendation_context += rec_text + "\n"
        
        chat_history = await asyncio.to_thread(get_chat_history, payload.session_id)
        
        full_prompt = f"""
        {SYSTEM_INSTRUCTION}

        ### Metadata: Score={payload.student_score}, Gov={payload.student_gov}, Track={payload.track}
        ### Context: {sanitize_llm_input(general_context, 500)}
        ### Top Recommendations: {recommendation_context}
        ### History: {chat_history}
        ### Question: {sanitize_llm_input(payload.question)}
        """

        try:
            response = await ai_client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=800,
                temperature=0.2
            )
            ai_response = response.choices[0].message.content
        except Exception as api_err:
            logger.error(f"Groq API Error: {api_err}")
            raise api_err
        if faculty_url and not pd.isna(faculty_url) and payload.interests:
            ai_response += f"\n\n🔗 [الموقع الرسمي للكلية]({faculty_url})"

        await asyncio.to_thread(save_chat, payload.session_id, payload.question, ai_response)
        return {"status": "success", "answer": ai_response, "wishes_75": wishes_75_list}
    except Exception as e:
        logger.error(f"Inference Error: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Internal server error.") from e