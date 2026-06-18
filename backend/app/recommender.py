"""
recommender.py — Hybrid Recommendation Engine for Tansiq.
Builds an ordered list of exactly 75 academic wishes (desires)
based on score eligibility, track matching, geographic zoning,
student interests, distance sorting, and priority weighting.
"""

import logging
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

def normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    return str(text).replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").strip()

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

    def apply_refinements(self, wishes_list: list[dict], refinements: list[dict], student_gov: str) -> list[dict]:
        if not refinements or not wishes_list:
            return wishes_list
        
        for ref in refinements:
            action = ref.get("action")
            
            if action == "inject":
                target_index = int(ref.get("target_index", len(wishes_list)))
                college_field = ref.get("college_field", "")
                
                norm_field = normalize(college_field).lower()
                available = self.df[self.df["قطاع الكلية"].apply(
                    lambda x, nf=norm_field: nf in normalize(str(x)).lower() or normalize(str(x)).lower() in nf if pd.notna(x) else False
                )].copy()
                if available.empty:
                    continue
                
                available["_dist"] = available["Governorate"].apply(lambda gov: self._get_distance(student_gov, gov))
                available = available.sort_values(by=["_dist", "Score"], ascending=[True, False])
                
                # Create a set of faculty names that are being targeted for injection
                target_faculty_names = {r["Faculty"] for _, r in available.iterrows()}
                
                # Extract clean list to inject
                to_inject = []
                for _, r in available.iterrows():
                    row_dict = r.to_dict()
                    row_dict["_distance"] = row_dict["_dist"]
                    to_inject.append(row_dict)
                
                # FORCE SAFETY COMPLIANCE: Remove targeted faculties from wishes_list first if they already exist lower down
                wishes_list = [w for w in wishes_list if w["Faculty"] not in target_faculty_names]
                
                # Perform the pristine slice insertion
                wishes_list = wishes_list[:target_index] + to_inject + wishes_list[target_index:]
                wishes_list = wishes_list[:TARGET_WISHES]
                
            elif action == "swap":
                old_faculty = ref.get("old_faculty", "")
                new_faculty = ref.get("new_faculty", "")
                
                new_row = self.df[self.df["Faculty"] == new_faculty]
                if new_row.empty:
                    continue
                
                new_row_dict = new_row.iloc[0].to_dict()
                new_row_dict["_distance"] = self._get_distance(student_gov, new_row_dict.get("Governorate", ""))
                
                for i, w in enumerate(wishes_list):
                    if w["Faculty"] == old_faculty:
                        new_row_dict["_tier"] = float(w.get("_tier", 99.0))
                        wishes_list[i] = new_row_dict
                        break
                        
        return wishes_list

    def recommend(
        self,
        student_score: float,
        student_gender: str,
        student_gov: str,
        track: str,
        interests: list[str] = None,
        priority: str = "مجموع", # pylint: disable=unused-argument
        refinements: list[dict] = None
    ) -> list[dict]:
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

        # Step 7 — Assign hierarchical tiers
        eligible["_tier"] = eligible.apply(self._assign_tier, axis=1)

        # Step 8 — Strict Sorting: Tier (Asc), Distance (Asc), Score (Desc)
        eligible = eligible.sort_values(
            by=["_tier", "_distance", "Score"],
            ascending=[True, True, False],
        )

        # Step 9 — Trim to exactly 75 wishes
        result_df = eligible.head(TARGET_WISHES).reset_index(drop=True)

        result_df = result_df.drop(
            columns=["_zone", "_distance", "_interest_score", "_tier", "_dist"], errors="ignore"
        )
        
        wishes_list = result_df.to_dict('records')

        if refinements:
            wishes_list = self.apply_refinements(wishes_list, refinements, student_gov)

        logger.info(
            "Generated %d wishes for student (score=%.1f, gov=%s, track=%s)",
            len(wishes_list), student_score, student_gov, track
        )
        return wishes_list

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

        college_field = str(row.get("قطاع الكلية", "")).strip()
        if not college_field:
            return 0.0

        score = 0.0

        # Generic mapping: Student Interest -> Supabase college_field categories
        INTEREST_MAPPING = {
            "برمجة": {
                "super_target": ["حاسبات", "ذكاء اصطناعي", "الذكاء الاصطناعي", "حاسبات ومعلومات"],
                "target": ["هندسة"],
                "related": ["علوم"]
            },
            "هندسة": {
                "super_target": ["هندسة"],
                "target": ["حاسبات", "ذكاء اصطناعي", "الذكاء الاصطناعي", "حاسبات ومعلومات"],
                "related": ["علوم"]
            },
            "طب": {
                "super_target": ["طب", "الطب"],
                "target": ["طب أسنان", "صيدلة", "علاج طبيعي", "الصيدلة"],
                "related": ["طب بيطري", "تمريض", "علوم صحية", "التمريض"]
            },
            "لغات": {
                "super_target": ["ألسن", "الألسن"],
                "target": ["لغات وترجمة"],
                "related": ["آداب", "تربية", "الآداب", "التربية"]
            },
            "فنون": {
                "super_target": ["فنون جميلة", "الفنون الجميلة", "فنون تطبيقية"],
                "target": [],
                "related": ["تربية فنية", "تربية نوعية", "التربية النوعية"]
            }
        }

        college_field_norm = normalize(college_field.lower())

        for index, interest in enumerate(interests):
            interest_clean = normalize(interest.lower())
            penalty = min(0.1 * index, 0.4)
            
            is_medical = "طب" in interest_clean or "صيدل" in interest_clean or "علاج" in interest_clean or "اسنان" in interest_clean
            if is_medical:
                if "بشر" in college_field_norm or college_field_norm == "طب":
                    score = max(score, 2.5 - penalty)
                elif "اسنان" in college_field_norm or "صيدل" in college_field_norm:
                    score = max(score, 2.0 - penalty)
                elif "علاج طبيعي" in college_field_norm or "علوم صحية" in college_field_norm or "علوم طبية" in college_field_norm:
                    score = max(score, 1.5 - penalty)
                
            matched_mapping = False
            for key, mapping in INTEREST_MAPPING.items():
                if key == "طب":
                    continue
                if key in interest_clean or interest_clean in key:
                    matched_mapping = True
                    if any(normalize(t) in college_field_norm or college_field_norm in normalize(t) for t in mapping.get("super_target", [])):
                        score = max(score, 2.5 - penalty)
                    elif any(normalize(t) in college_field_norm or college_field_norm in normalize(t) for t in mapping.get("target", [])):
                        score = max(score, 2.0 - penalty)
                    elif any(normalize(r) in college_field_norm or college_field_norm in normalize(r) for r in mapping.get("related", [])):
                        score = max(score, 1.5 - penalty)
                    
            if not matched_mapping:
                # Dynamic fallback: bidirectional flexible substring match
                if interest_clean in college_field_norm or college_field_norm in interest_clean:
                    score = max(score, 2.0 - penalty)

        return score

    @staticmethod
    def _assign_tier(row: pd.Series) -> float:
        interest_score = row.get("_interest_score", 0.0)
        zone = row.get("_zone", 2)  # 0 for A, 1 for B, 2 for C
        faculty_name = str(row.get("Faculty", "")).strip()
        college_field = normalize(str(row.get("قطاع الكلية", "")).lower())

        # Penalize private institutes: push them below ALL government faculties
        is_institute = any(word in faculty_name for word in ["معهد", "العالى", "عالي", "أكاديمية", "اكاديمية"])
        offset = 12 if is_institute else 0

        # Strict Base Tier Segmentation based cleanly on the row's normalized category
        if "بشر" in college_field or college_field == "طب":
            base_tier = 1 + zone
        elif "اسنان" in college_field:
            base_tier = 4 + zone
        elif "صيدل" in college_field:
            base_tier = 7 + zone
        elif "علاج طبيعي" in college_field:
            base_tier = 10 + zone
        elif "بيطري" in college_field or "علوم صحية" in college_field:
            base_tier = 13 + zone
        elif "حاسبات" in college_field or "ذكاء" in college_field:
            base_tier = 16 + zone
        else:
            base_tier = 19 + zone

        # Scale the interest_score so it acts as an internal sub-tier tiebreaker 
        # without ever blurring or overlapping the strict integer boundaries.
        return float(base_tier - (interest_score / 10.0) + offset)


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
        f"الرؤية: {best_row.get('رؤية الكلية', '')}. "
    )
    
    depts = best_row.get('general_departments', '')
    if pd.notna(depts) and depts:
        context_text += f"الأقسام العامة: {depts}. "
        
    sections = best_row.get('special_sections', '')
    if pd.notna(sections) and sections:
        context_text += f"الأقسام الخاصة: {sections}. "

    url = best_row.get("URL", None)
    if pd.isna(url):
        url = None

    return {"text": context_text, "url": url}
