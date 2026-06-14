"""
recommender.py — Hybrid Recommendation Engine for Tansiq.
Builds an ordered list of exactly 75 academic wishes (desires)
based on score eligibility, track matching, geographic zoning,
student interests, distance sorting, and priority weighting.
"""

import logging
import re
from difflib import SequenceMatcher

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geographic Zone Mapping (Fallback if DB match fails)
# ---------------------------------------------------------------------------
_ZONE_MAP: dict[str, list[str]] = {
    "القاهرة": ["الجيزة", "القليوبية", "الشرقية", "المنوفية"],
    "الجيزة": ["القاهرة", "القليوبية", "الفيوم", "بني سويف", "المنيا"],
    "الإسكندرية": ["البحيرة", "مطروح", "الغربية", "كفر الشيخ"],
    "الدقهلية": ["دمياط", "الشرقية", "الغربية", "كفر الشيخ", "القليوبية"],
    "الشرقية": ["القاهرة", "القليوبية", "الإسماعيلية", "الدقهلية", "المنوفية"],
    "القليوبية": ["القاهرة", "الجيزة", "المنوفية", "الشرقية", "الغربية"],
    "الغربية": ["المنوفية", "كفر الشيخ", "الدقهلية", "البحيرة", "القليوبية"],
    "المنوفية": ["القاهرة", "القليوبية", "الغربية", "الجيزة", "البحيرة"],
    "البحيرة": ["الإسكندرية", "الغربية", "كفر الشيخ", "المنوفية"],
    "كفر الشيخ": ["الغربية", "البحيرة", "الدقهلية", "الإسكندرية"],
    "دمياط": ["الدقهلية", "بورسعيد", "الشرقية"],
    "بورسعيد": ["دمياط", "الإسماعيلية", "الشرقية"],
    "الإسماعيلية": ["بورسعيد", "الشرقية", "السويس"],
    "السويس": ["الإسماعيلية", "القاهرة", "البحر الأحمر"],
    "الفيوم": ["الجيزة", "بني سويف", "المنيا"],
    "بني سويف": ["الفيوم", "الجيزة", "المنيا"],
    "المنيا": ["بني سويف", "أسيوط", "الفيوم"],
    "أسيوط": ["المنيا", "سوهاج", "الوادي الجديد"],
    "سوهاج": ["أسيوط", "قنا", "الوادي الجديد"],
    "قنا": ["سوهاج", "الأقصر", "البحر الأحمر"],
    "الأقصر": ["قنا", "أسوان", "البحر الأحمر"],
    "أسوان": ["الأقصر", "البحر الأحمر"],
    "البحر الأحمر": ["السويس", "قنا", "الأقصر", "أسوان"],
    "مطروح": ["الإسكندرية", "البحيرة"],
    "الوادي الجديد": ["أسيوط", "سوهاج", "المنيا"],
    "شمال سيناء": ["الإسماعيلية", "جنوب سيناء", "بورسعيد"],
    "جنوب سيناء": ["السويس", "شمال سيناء", "البحر الأحمر"],
}

TARGET_WISHES = 75

class HybridRecommender:
    def __init__(
        self,
        df: pd.DataFrame,
        df_geo: pd.DataFrame,
        df_dist: pd.DataFrame,
    ):
        self.df = df.copy()
        self.df_geo = df_geo
        self.df_dist = df_dist

    def recommend(
        self,
        student_score: float,
        student_gender: str,
        student_gov: str,
        track: str,
        interests: list[str] | None = None,
        priority: str = "غير محدد",
    ) -> pd.DataFrame:
        interests = interests or []

        # Step 1 — Filter by score
        eligible = self.df[self.df["Score"] <= student_score].copy()

        if eligible.empty:
            logger.warning("No faculties found for score %.1f", student_score)
            return pd.DataFrame()

        # Step 2 — Filter by gender
        eligible = self._filter_gender(eligible, student_gender)

        # Step 3 — Filter by track using DB flags (is_science, is_math, is_arts)
        eligible = self._filter_track(eligible, track)

        if eligible.empty:
            return pd.DataFrame()

        # Step 4 — Assign geographic zones (0=A, 1=B, 2=C)
        eligible["_zone"] = eligible["Governorate"].apply(
            lambda gov: self._get_zone(student_gov, gov)
        )

        # Step 5 — Map Distances
        eligible["_distance"] = eligible["Governorate"].apply(
            lambda gov: self._get_distance(student_gov, gov)
        )

        # Step 6 — Interest boost score
        eligible["_interest_score"] = eligible.apply(
            lambda row: self._calc_interest_score(row, interests), axis=1
        )

        # Step 7 — Sort based on priority, zone, distance
        eligible = self._apply_priority_sort(eligible, priority)

        # Step 8 — Trim to exactly 75 wishes
        result = eligible.head(TARGET_WISHES).reset_index(drop=True)

        result = result.drop(
            columns=["_zone", "_distance", "_interest_score"], errors="ignore"
        )

        logger.info(
            "Generated %d wishes for student (score=%.1f, gov=%s, track=%s)",
            len(result), student_score, student_gov, track
        )
        return result

    @staticmethod
    def _filter_gender(df: pd.DataFrame, gender: str) -> pd.DataFrame:
        gender_lower = gender.strip()
        if gender_lower in ("ذكر", "بنين", "male"):
            if "بنين" in df.columns:
                return df[df["بنين"].astype(str).str.strip() != "0"]
        if gender_lower in ("أنثى", "انثى", "بنات", "female"):
            if "بنات" in df.columns:
                return df[df["بنات"].astype(str).str.strip() != "0"]
        return df

    @staticmethod
    def _filter_track(df: pd.DataFrame, track: str) -> pd.DataFrame:
        track_clean = track.strip()
        if "علمي علوم" in track_clean and "is_science" in df.columns:
            return df[df["is_science"] == 1]
        elif "علمي رياضة" in track_clean and "is_math" in df.columns:
            return df[df["is_math"] == 1]
        elif "أدبي" in track_clean or "ادبي" in track_clean:
            if "is_arts" in df.columns:
                return df[df["is_arts"] == 1]
        return df

    def _get_zone(self, student_gov: str, faculty_gov: str) -> int:
        student_gov = student_gov.strip()
        faculty_gov = faculty_gov.strip()

        if student_gov == faculty_gov:
            return 0  # Zone A

        # Try to resolve via df_geo
        # df_geo maps 'educational_administration' to 'university_name' and 'category'
        if not self.df_geo.empty and "category" in self.df_geo.columns:
            try:
                matches = self.df_geo[
                    self.df_geo["educational_administration"].astype(str).str.contains(student_gov, na=False, regex=False)
                ]
                if not matches.empty:
                    # We check if faculty_gov matches the university_name roughly
                    # Since faculty_gov is a governorate, and university_name has the gov name often
                    univ_match = matches[matches["university_name"].astype(str).str.contains(faculty_gov, na=False, regex=False)]
                    if not univ_match.empty:
                        cat = str(univ_match.iloc[0]["category"]).strip()
                        if cat == 'أ': return 0
                        if cat == 'ب': return 1
                        if cat == 'ج': return 2
            except Exception:
                pass

        # Fallback to predefined map
        neighbors = _ZONE_MAP.get(student_gov, [])
        if faculty_gov in neighbors:
            return 1  # Zone B
        return 2  # Zone C

    def _get_distance(self, student_gov: str, faculty_gov: str) -> float:
        """Fetch distance between student gov and faculty gov from df_dist."""
        if student_gov == faculty_gov:
            return 0.0

        if not self.df_dist.empty and "distance_km" in self.df_dist.columns:
            try:
                # Filter by origin_governorate
                matches = self.df_dist[
                    self.df_dist["origin_governorate"].astype(str).str.contains(student_gov, na=False, regex=False)
                ]
                if not matches.empty:
                    # Filter by destination_university (which usually contains the governorate name)
                    dest_match = matches[
                        matches["destination_university"].astype(str).str.contains(faculty_gov, na=False, regex=False)
                    ]
                    if not dest_match.empty:
                        return float(dest_match.iloc[0]["distance_km"])
            except Exception:
                pass

        # Fallback default high distance if not found
        return 9999.0

    @staticmethod
    def _calc_interest_score(row: pd.Series, interests: list[str]) -> float:
        if not interests:
            return 0.0

        score = 0.0
        searchable_text = " ".join(
            str(row.get(col, ""))
            for col in ["Faculty", "رؤية الكلية", "قطاع الكلية", "general_departments"]
        ).lower()

        for interest in interests:
            interest_lower = interest.lower().strip()
            if interest_lower in searchable_text:
                score += 2.0
            elif any(
                SequenceMatcher(None, interest_lower, word).ratio() > 0.6
                for word in searchable_text.split()
            ):
                score += 1.0

        return score

    @staticmethod
    def _apply_priority_sort(df: pd.DataFrame, priority: str) -> pd.DataFrame:
        priority = priority.strip()

        if priority in ("محافظة", "الجغرافيا"):
            # 1. Zone (A -> B -> C)
            # 2. Distance (Closest first)
            # 3. Interest
            # 4. Score
            return df.sort_values(
                by=["_zone", "_distance", "_interest_score", "Score"],
                ascending=[True, True, False, False],
            )

        if priority in ("تخصص", "الكلية"):
            # 1. Interest
            # 2. Score
            # 3. Zone
            # 4. Distance
            return df.sort_values(
                by=["_interest_score", "Score", "_zone", "_distance"],
                ascending=[False, False, True, True],
            )

        # Default
        return df.sort_values(
            by=["_zone", "_distance", "_interest_score", "Score"],
            ascending=[True, True, False, False],
        )

def search_context(df: pd.DataFrame, question: str) -> dict[str, str | None]:
    if df.empty or not question:
        return {"text": "", "url": None}

    question_lower = question.lower().strip()
    best_score = 0.0
    best_row = None

    for _, row in df.iterrows():
        faculty_name = str(row.get("Faculty", "")).lower()
        vision = str(row.get("رؤية الكلية", "")).lower()
        sector = str(row.get("قطاع الكلية", "")).lower()
        combined = f"{faculty_name} {vision} {sector}"

        match_score = 0.0
        for word in question_lower.split():
            if len(word) < 2:
                continue
            if word in combined:
                match_score += 2.0
            elif any(SequenceMatcher(None, word, w).ratio() > 0.65 for w in combined.split() if len(w) > 2):
                match_score += 0.5

        if match_score > best_score:
            best_score = match_score
            best_row = row

    if best_row is None or best_score < 1.0:
        return {"text": "", "url": None}

    context_text = (
        f"كلية {best_row.get('Faculty', '')} "
        f"في محافظة {best_row.get('Governorate', '')}. "
        f"الحد الأدنى: {best_row.get('Score', 'غير متاح')}. "
        f"القطاع: {best_row.get('قطاع الكلية', '')}. "
        f"الرؤية: {best_row.get('رؤية الكلية', '')}"
    )

    url = best_row.get("URL", None)
    if pd.isna(url):
        url = None

    return {"text": context_text, "url": url}
