import logging
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

from app.config import settings
from app.database import load_initial_data, get_chat_history, save_chat, upsert_student_profile, get_student_profile
from app.recommender import HybridRecommender
from app.scraper import smart_lookup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Tansiq ML Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DataOps: Load data pipelines during startup
df, df_geo, df_dist = load_initial_data()
ai_client = genai.Client(api_key=settings.GEMINI_API_KEY)

class InferenceRequest(BaseModel):
    session_id: str  
    question: Optional[str] = ""
    student_score: Optional[float] = None
    student_gender: Optional[str] = None
    student_gov: Optional[str] = None
    track: Optional[str] = None
    interests: Optional[List[str]] = []
    priority: Optional[str] = "غير محدد"

SYSTEM_INSTRUCTION = """You are "Tansiq Assistant", a professional academic advisor helping Egyptian high school students. 
Tone: Friendly Egyptian Arabic (عامية مصرية ودية). Close the answer with "بالتوفيق يا بطل!" and halt."""

@app.post("/predict")
def predict_answer(request: InferenceRequest):
    try:
        # Check profile variables and upsert/fallback
        if request.student_score is not None and request.track is not None:
            upsert_student_profile(
                request.session_id, request.student_score, request.student_gender or "", 
                request.student_gov or "", request.track, request.interests or [], request.priority or ""
            )
        else:
            profile = get_student_profile(request.session_id)
            if profile:
                request.student_score = request.student_score or profile.get("student_score")
                request.student_gender = request.student_gender or profile.get("student_gender")
                request.student_gov = request.student_gov or profile.get("student_gov")
                request.track = request.track or profile.get("track")
                request.interests = request.interests or profile.get("interests", [])
                request.priority = request.priority or profile.get("priority")

        # Smart context lookup
        search_result = smart_lookup(request.question)
        general_context = ""
        faculty_url = None
        if search_result["results"]:
            top_result = search_result["results"][0]
            general_context = top_result.get("text", "") or str(top_result)
            faculty_url = top_result.get("url")
        
        recommender = HybridRecommender(df, df_geo, df_dist)
        
        # Guard clause for missing vital info
        if request.student_score is None or not request.track:
            missing_info_response = "أهلاً بيك يا بطل! عشان أقدر أساعدك بشكل دقيق، محتاج أعرف مجموعك كام في الثانوية العامة وإنت قسم إيه (علمي علوم ولا رياضة ولا أدبي)؟"
            save_chat(request.session_id, request.question, missing_info_response)
            return {"status": "success", "answer": missing_info_response, "wishes_75": []}

        recommendations = recommender.recommend(
            student_score=request.student_score, student_gender=request.student_gender or "",
            student_gov=request.student_gov or "", track=request.track,
            interests=request.interests or [], priority=request.priority or "غير محدد"
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
        
        if not request.interests or request.priority not in ["تخصص", "محافظة"]:
            fast_response = "أنا سعيد جدا إني بساعدك! مجموعك ما شاء الله ممتاز ومتاح ليك خيارات كتير.\nبس عشان أقدر أرتبلك الكليات بشكل دقيق يفيدك، محتاج أسألك سؤال واحد:\nهل الأهم عندك تدخل الكلية اللي بتحبها حتى لو في محافظة تانية، ولا الأهم تفضل في محافظتك؟"
            save_chat(request.session_id, request.question, fast_response)
            return {"status": "success", "answer": fast_response, "wishes_75": wishes_75_list}
            
        recommendation_context = ""
        if not recommendations.empty:
            for _, row in recommendations.head(5).iterrows():
                recommendation_context += f"- كلية {row.get('Faculty')} ({row.get('Governorate')}) تقبل من {row.get('Score')}\n"
        
        priority_text = "تفضل في محافظتك" if request.priority == "محافظة" else "تدخل التخصص اللي بتحبه"
        chat_history = get_chat_history(request.session_id)
        
        user_prompt = f"""
        ### Metadata: Score={request.student_score}, Gov={request.student_gov}, Track={request.track}, Priority={priority_text}
        ### Context: {general_context}
        ### Top Recommendations: {recommendation_context}
        ### History: {chat_history}
        ### Question: {request.question}
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
        if faculty_url and not pd.isna(faculty_url) and request.interests:
             ai_response += "\n\n🔗 للمزيد من التفاصيل، تفضل بزيارة الموقع الرسمي للكلية."

        save_chat(request.session_id, request.question, ai_response)
        return {"status": "success", "answer": ai_response, "wishes_75": wishes_75_list}
    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")