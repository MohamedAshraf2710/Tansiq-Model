import sys
import pandas as pd
from app.recommender import HybridRecommender
from app.database import load_initial_data

def test_recommendation():
    df, df_geo, df_dist = load_initial_data()
    recommender = HybridRecommender(df, df_geo, df_dist)
    
    recs = recommender.recommend(
        student_score=380.0,
        student_gender="ذكر",
        student_gov="القاهرة",
        track="علمي رياضة",
        interests=["برمجة"],
    )
    
    recs = recs.head(20)
    with open("debug_output_2.md", "w", encoding="utf-8") as f:
        f.write("| Faculty | Governorate | Score | Tier | Dist | IntScore |\n")
        f.write("|---------|-------------|-------|------|------|----------|\n")
        for _, row in recs.iterrows():
            f.write(f"| {row['Faculty']} | {row['Governorate']} | {row['Score']} | {row.get('_tier')} | {row.get('_distance')} | {row.get('_interest_score')} |\n")

if __name__ == "__main__":
    test_recommendation()
