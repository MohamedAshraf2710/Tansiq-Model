import logging
import json
import pandas as pd
from sqlalchemy import create_engine, text
from app.config import settings

logger = logging.getLogger(__name__)
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

def load_initial_data():
    try:
        df = pd.read_sql_query("SELECT * FROM faculties", engine)
        df.rename(columns={
            'faculty': 'Faculty', 'governorate': 'Governorate', 'score': 'Score',
            'college_vision': 'رؤية الكلية', 'url': 'URL', 'college_field': 'قطاع الكلية',
            'boys': 'بنين', 'girls': 'بنات'
        }, inplace=True)
        
        df_geo = pd.read_sql_query("SELECT * FROM geo_distribution", engine)
        df_dist = pd.read_sql_query("SELECT * FROM university_distances", engine)
        
        logger.info("DataOps: Pipeline data successfully pulled from Supabase.")
        return df, df_geo, df_dist
    except Exception as e:
        logger.error(f"DataOps Pipeline Failed: {type(e).__name__}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def get_chat_history(session_id: str) -> str:
    if not session_id: return ""
    try:
        with engine.connect() as conn:
            query = text("SELECT user_message, ai_response FROM chat_history WHERE session_id = :session_id ORDER BY created_at ASC LIMIT 5")
            result = conn.execute(query, {"session_id": session_id})
            return "".join([f"Student: {row[0]}\nAdvisor: {row[1]}\n---\n" for row in result])
    except Exception:
        return ""

def save_chat(session_id: str, user_msg: str, ai_msg: str):
    if not session_id: return
    try:
        with engine.begin() as conn:  
            query = text("INSERT INTO chat_history (session_id, user_message, ai_response) VALUES (:session_id, :user_msg, :ai_msg)")
            conn.execute(query, {"session_id": session_id, "user_msg": user_msg, "ai_msg": ai_msg})
    except Exception:
        pass

def upsert_student_profile(session_id: str, score: float, gender: str, gov: str, track: str, interests: list, priority: str):
    if not session_id: return
    try:
        interests_json = json.dumps(interests)
        with engine.begin() as conn:
            query = text("""
                INSERT INTO student_profiles (session_id, student_score, student_gender, student_gov, track, interests, priority)
                VALUES (:session_id, :score, :gender, :gov, :track, :interests, :priority)
                ON CONFLICT (session_id) DO UPDATE SET
                    student_score = EXCLUDED.student_score,
                    student_gender = EXCLUDED.student_gender,
                    student_gov = EXCLUDED.student_gov,
                    track = EXCLUDED.track,
                    interests = EXCLUDED.interests,
                    priority = EXCLUDED.priority
            """)
            conn.execute(query, {
                "session_id": session_id, "score": score, "gender": gender, 
                "gov": gov, "track": track, "interests": interests_json, "priority": priority
            })
    except Exception as e:
        logger.error(f"Upsert Student Profile Failed: {type(e).__name__}")

def get_student_profile(session_id: str) -> dict:
    if not session_id: return {}
    try:
        with engine.connect() as conn:
            query = text("SELECT student_score, student_gender, student_gov, track, interests, priority FROM student_profiles WHERE session_id = :session_id")
            result = conn.execute(query, {"session_id": session_id}).fetchone()
            if result:
                return {
                    "student_score": result[0],
                    "student_gender": result[1],
                    "student_gov": result[2],
                    "track": result[3],
                    "interests": json.loads(result[4]) if result[4] else [],
                    "priority": result[5]
                }
    except Exception as e:
        logger.error(f"Get Student Profile Failed: {type(e).__name__}")
    return {}