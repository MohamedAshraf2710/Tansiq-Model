import pandas as pd

def search_context(df: pd.DataFrame, query: str):
    if df.empty: 
        return {"text": "", "url": None}
    ignore_words = ['تنسيق', 'كلية', 'ماهو', 'ماذا', 'كام', 'مجموع', 'معلومات', 'عن', 'ترشحلي', 'حاجة']
    words = [w for w in query.split() if len(w) > 2 and w not in ignore_words]
    
    results = df.copy()
    for word in words:
        mask = results['Faculty'].str.contains(word, case=False, na=False) | \
               results['Governorate'].str.contains(word, case=False, na=False)
        if not results[mask].empty:
            results = results[mask]
            
    if results.empty:
        return {"text": "", "url": None}
    
    row = results.iloc[0]
    return {
        "text": f"Confirmed fact from Tansiq Office: {row.get('Faculty')} located in {row.get('Governorate')} governorate requires a minimum score of {row.get('Score')}.",
        "url": row.get('URL', None)
    }

class HybridRecommender:
    def __init__(self, data_frame: pd.DataFrame, geo_df: pd.DataFrame, dist_df: pd.DataFrame):
        self.df = data_frame.copy()
        self.geo_df = geo_df
        self.dist_df = dist_df

    def get_real_distance(self, origin_gov, dest_gov):
        if not self.dist_df.empty:
            match = self.dist_df[(self.dist_df['origin_governorate'] == origin_gov) & (self.dist_df['destination_university'].str.contains(dest_gov, na=False))]
            if not match.empty:
                return match.iloc[0]['distance_km']
        return 0 if origin_gov == dest_gov else 500

    def recommend(self, student_score, student_gender, student_gov, interests, track, priority):
        if self.df.empty: return pd.DataFrame()
        results = self.df.copy()

        # 1. Base Filtering
        results = results[results['Score'] <= student_score]
        if student_gender == 'ذكر' and 'بنين' in results.columns: results = results[results['بنين'] == 1]
        elif student_gender == 'أنثى' and 'بنات' in results.columns: results = results[results['بنات'] == 1]
        
        if track == "علمي علوم" and 'is_science' in results.columns: results = results[results['is_science'] == 1]
        elif track == "علمي رياضة" and 'is_math' in results.columns: results = results[results['is_math'] == 1]
        elif track == "أدبي" and 'is_arts' in results.columns: results = results[results['is_arts'] == 1]

        if results.empty: return pd.DataFrame()

        # 2. Advanced Scoring System
        def calculate_score(row):
            total_points = float(row.get('Score', 0))
            faculty_name = str(row.get('Faculty', ''))
            sector = str(row.get('قطاع الكلية', ''))
            gov = str(row.get('Governorate', ''))
            
            geo_category = 'ج'
            if not self.geo_df.empty:
                user_geo = self.geo_df[self.geo_df['educational_administration'].str.contains(student_gov, na=False)]
                for _, geo_row in user_geo.iterrows():
                    uni_name = str(geo_row['university_name']).replace('جامعة ', '').strip()
                    if uni_name in faculty_name or uni_name == gov:
                        geo_category = str(geo_row['category']).strip()
                        break
            
            if geo_category == 'أ': total_points += 50000  
            elif geo_category == 'ب': total_points += 30000  
            else: total_points += 10000  

            if priority == "تخصص":
                for interest in interests:
                    if interest in faculty_name or interest in sector: total_points += 100000 
                if gov == student_gov or geo_category == 'أ': total_points += 5000 
            elif priority == "محافظة":
                if gov == student_gov or geo_category == 'أ': total_points += 100000 
                for interest in interests:
                    if interest in faculty_name or interest in sector: total_points += 5000 
            else:
                for interest in interests:
                    if interest in faculty_name or interest in sector: total_points += 10000
                if gov == student_gov or geo_category == 'أ': total_points += 10000
                
            return total_points

        results['Recommendation_Score'] = results.apply(calculate_score, axis=1)
        results['Distance_KM'] = results.apply(lambda x: self.get_real_distance(student_gov, x['Governorate']), axis=1)
        
        final_results = results.sort_values(
            by=['Recommendation_Score', 'Distance_KM'], 
            ascending=[False, True]
        ).head(75)
        
        return final_results