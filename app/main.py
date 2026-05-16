import logging
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

from app.config import settings
from app.database import load_initial_data, get_chat_history, save_chat
from app.recommender import HybridRecommender, search_context
from app.scraper import scrape_faculty_details

# Setup Logging Pipeline
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Tansiq ML Microservice")

# DataOps: Load data pipelines during startup phase
df, df_geo, df_dist = load_initial_data()
ai_client = genai.Client(api_key=settings.GEMINI_API_KEY)

class InferenceRequest(BaseModel):
    session_id: str  
    question: str
    student_score: float
    student_gender: str
    student_gov: str
    track: str
    interests: Optional[List[str]] = []
    priority: Optional[str] = "غير محدد"

SYSTEM_INSTRUCTION = """You are "Tansiq Assistant", a professional academic advisor helping Egyptian high school students. 
Tone: Friendly Egyptian Arabic (عامية مصرية ودية). Close the answer with "بالتوفيق يا بطل!" and halt."""

@app.post("/predict")
def predict_answer(request: InferenceRequest):
    try:
        logger.info(f"Processing inference request for session: {request.session_id}")
        
        # 1. Search deterministic database context
        search_result = search_context(df, request.question)
        general_context = search_result.get("text", "")
        faculty_url = search_result.get("url", None)
        
        # 2. Extract best recommendation using Hybrid Heuristics
        recommender = HybridRecommender(df, df_geo, df_dist)
        recommendations = recommender.recommend(
            student_score=request.student_score, 
            student_gender=request.student_gender,
            student_gov=request.student_gov, 
            track=request.track,
            interests=request.interests, 
            priority=request.priority
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
        
        # Rule-based bypass routing for incomplete student priority/metadata
        if not request.interests or request.priority not in ["تخصص", "محافظة"]:
            fast_response = "أنا سعيد جدا إني بساعدك! مجموعك ما شاء الله ممتاز ومتاح ليك خيارات كتير.\nبس عشان أقدر أرتبلك الكليات بشكل دقيق يفيدك، محتاج أسألك سؤال واحد:\nهل الأهم عندك تدخل الكلية اللي بتحبها حتى لو في محافظة تانية، ولا الأهم تفضل في محافظتك؟"
            save_chat(request.session_id, request.question, fast_response)
            return {"status": "success", "answer": fast_response, "wishes_75": wishes_75_list}
            
        # 3. Enrich context from Database extended knowledge base
        enriched_db_context = ""
        recommendation_context = ""
        
        if not recommendations.empty:
            top_3 = recommendations.head(3)
            for _, row in top_3.iterrows():
                recommendation_context += f"- كلية {row.get('Faculty')} ({row.get('Governorate')}) تقبل من {row.get('Score')}\n"
                if pd.notna(row.get('رؤية الكلية')):
                    enriched_db_context += f"رؤية وأهداف {row.get('Faculty')}: {row.get('رؤية الكلية')}\n"
        
        # 4. Fallback Web Scraper Activation Pattern
        scraped_context = ""
        if faculty_url and any(word in request.question for word in ["تفاصيل", "موقع", "معلومات أكثر", "أقسام"]):
            scraped_context = scrape_faculty_details(faculty_url)
            if scraped_context:
                scraped_context = f"\nمعلومات إضافية حية من موقع الكلية: {scraped_context}"

        priority_text = "تفضل في محافظتك" if request.priority == "محافظة" else "تدخل التخصص اللي بتحبه"
        chat_history = get_chat_history(request.session_id)
        
        # Structuring Context Prompt Payload without code alignment tabs
        user_prompt = (
            f"### Metadata: Score={request.student_score}, Gov={request.student_gov}, Track={request.track}, Priority={priority_text}\n"
            f"### Core Context: {general_context}\n"
            f"### Database Extended Insights: {enriched_db_context} {scraped_context}\n"
            f"### Top Recommendations:\n{recommendation_context}\n"
            f"### History:\n{chat_history}\n"
            f"### Question: {request.question}"
        )

        # 5. Execute Non-Hallucinating LLM Inference Generation
        response = ai_client.models.generate_content(
            model=settings.MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=500  
            )
        )
        
        ai_response = response.text
        
        # Inject explicit Official URL redirection call-to-action
        if faculty_url and not pd.isna(faculty_url):
             ai_response += f"\n\nللمزيد من التفاصيل، تفضل بزيارة الموقع الرسمي للكلية: {faculty_url}"

        save_chat(request.session_id, request.question, ai_response)
        logger.info(f"Successfully served request for session: {request.session_id}")
        return {"status": "success", "answer": ai_response, "wishes_75": wishes_75_list}

    except Exception as e:
        logger.error(f"Critical Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred.")