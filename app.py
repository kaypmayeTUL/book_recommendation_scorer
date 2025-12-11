import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
import io

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    return True

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Library Book Recommendation Scorer",
    page_icon="📚",
    layout="wide"
)

# Header
st.title("📚 Library Book Recommendation Scorer")
st.markdown("""
This tool helps you **score and rank book recommendations** based on how well they match your library's checkout history.

### How it works:
1. Upload your **checkout history** (CSV with title, author, checkouts, subjects, LC classification)
2. Upload your **book recommendations** (CSV with title, author, subjects, LC classification)
3. Get scored and ranked recommendations based on:
   - Subject/Topic similarity (50%)
   - Checkout patterns (30%)
   - Author popularity (20%)
""")

# Recommendation Scorer Class (from the notebook)
class RecommendationScorer:
    def __init__(self, checkouts_df):
        self.checkouts_df = checkouts_df
        self.stemmer = SnowballStemmer('english')
        self.author_checkout_map = self._build_author_map()
        self.lc_checkout_map = self._build_lc_map()
        self.subject_terms = self._extract_subject_terms()
        
    def _build_author_map(self):
        author_map = defaultdict(list)
        for _, row in self.checkouts_df.iterrows():
            if pd.notna(row.get('author')):
                author_clean = self._clean_author(row['author'])
                if author_clean:
                    author_map[author_clean].append(row.get('checkouts', 0))
        return dict(author_map)
    
    def _build_lc_map(self):
        lc_map = defaultdict(list)
        for _, row in self.checkouts_df.iterrows():
            if pd.notna(row.get('lc_classification')):
                lc_prefix = self._get_lc_prefix(row['lc_classification'])
                if lc_prefix:
                    lc_map[lc_prefix].append(row.get('checkouts', 0))
        return dict(lc_map)
    
    def _extract_subject_terms(self):
        all_terms = []
        for _, row in self.checkouts_df.iterrows():
            if pd.notna(row.get('subjects')):
                subjects = str(row['subjects']).split(';')
                checkouts = row.get('checkouts', 0)
                for subject in subjects:
                    terms = self._tokenize_and_stem(subject)
                    all_terms.extend([(term, checkouts) for term in terms])
        
        term_checkouts = defaultdict(list)
        for term, checkout_count in all_terms:
            term_checkouts[term].append(checkout_count)
        
        return {term: sum(counts)/len(counts) for term, counts in term_checkouts.items()}
    
    def _clean_author(self, author):
        if pd.isna(author):
            return None
        author_clean = re.sub(r'[^\w\s]', '', str(author).lower())
        author_clean = ' '.join(author_clean.split())
        return author_clean if len(author_clean) > 2 else None
    
    def _get_lc_prefix(self, lc_class):
        if pd.isna(lc_class):
            return None
        lc_str = str(lc_class).strip().upper()
        match = re.match(r'^([A-Z]{1,3})', lc_str)
        return match.group(1) if match else None
    
    def _tokenize_and_stem(self, text):
        if pd.isna(text):
            return []
        text_clean = re.sub(r'[^\w\s]', ' ', str(text).lower())
        words = text_clean.split()
        stemmed = [self.stemmer.stem(word) for word in words if len(word) > 2]
        return stemmed
    
    def _get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(self.stemmer.stem(lemma.name().lower()))
        return synonyms
    
    def _calculate_subject_similarity(self, recommendation):
        if pd.isna(recommendation.get('subjects')) or not self.subject_terms:
            return 0.0
        
        rec_terms = self._tokenize_and_stem(recommendation['subjects'])
        if not rec_terms:
            return 0.0
        
        total_score = 0
        matched_terms = 0
        
        for rec_term in rec_terms:
            rec_syns = self._get_synonyms(rec_term)
            rec_syns.add(rec_term)
            
            max_term_score = 0
            for syn in rec_syns:
                if syn in self.subject_terms:
                    max_term_score = max(max_term_score, self.subject_terms[syn])
            
            if max_term_score > 0:
                matched_terms += 1
                total_score += max_term_score
        
        if matched_terms == 0:
            return 0.0
        
        avg_checkouts = total_score / matched_terms
        max_checkouts = max(self.subject_terms.values()) if self.subject_terms else 1
        
        coverage = matched_terms / len(rec_terms)
        normalized_score = (avg_checkouts / max_checkouts) * 100
        
        return normalized_score * (0.7 + 0.3 * coverage)
    
    def _calculate_lc_score(self, recommendation):
        if pd.isna(recommendation.get('lc_classification')) or not self.lc_checkout_map:
            return 0.0
        
        lc_prefix = self._get_lc_prefix(recommendation['lc_classification'])
        if not lc_prefix or lc_prefix not in self.lc_checkout_map:
            return 0.0
        
        checkouts = self.lc_checkout_map[lc_prefix]
        avg_checkouts = sum(checkouts) / len(checkouts)
        max_checkouts = max([sum(v)/len(v) for v in self.lc_checkout_map.values()])
        
        return (avg_checkouts / max_checkouts) * 100
    
    def _calculate_author_score(self, recommendation):
        if pd.isna(recommendation.get('author')) or not self.author_checkout_map:
            return 0.0
        
        author_clean = self._clean_author(recommendation['author'])
        if not author_clean or author_clean not in self.author_checkout_map:
            return 0.0
        
        checkouts = self.author_checkout_map[author_clean]
        avg_checkouts = sum(checkouts) / len(checkouts)
        max_checkouts = max([sum(v)/len(v) for v in self.author_checkout_map.values()])
        
        return (avg_checkouts / max_checkouts) * 100
    
    def score_recommendations(self, recommendations_df, 
                            subject_weight=0.5, 
                            lc_weight=0.3, 
                            author_weight=0.2):
        results = []
        
        for idx, rec in recommendations_df.iterrows():
            subject_score = self._calculate_subject_similarity(rec)
            lc_score = self._calculate_lc_score(rec)
            author_score = self._calculate_author_score(rec)
            
            likelihood_score = (
                subject_score * subject_weight +
                lc_score * lc_weight +
                author_score * author_weight
            )
            
            rec_dict = rec.to_dict()
            rec_dict['likelihood_score'] = round(likelihood_score, 2)
            rec_dict['similarity_score'] = round(subject_score, 2)
            rec_dict['checkout_volume_score'] = round(lc_score, 2)
            rec_dict['author_popularity_score'] = round(author_score, 2)
            
            results.append(rec_dict)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('likelihood_score', ascending=False)
        results_df = results_df.reset_index(drop=True)
        
        return results_df

# Helper function to generate text report
def generate_report(results_df):
    report = []
    report.append("=" * 80)
    report.append("LIBRARY BOOK RECOMMENDATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Total Recommendations Analyzed: {len(results_df)}")
    report.append("")
    
    high = len(results_df[results_df['likelihood_score'] >= 70])
    medium = len(results_df[(results_df['likelihood_score'] >= 40) & 
                           (results_df['likelihood_score'] < 70)])
    low = len(results_df[results_df['likelihood_score'] < 40])
    
    report.append(f"High Priority (70-100):   {high} books  ({high/len(results_df)*100:.1f}%)")
    report.append(f"Medium Priority (40-69):  {medium} books  ({medium/len(results_df)*100:.1f}%)")
    report.append(f"Low Priority (0-39):      {low} books  ({low/len(results_df)*100:.1f}%)")
    report.append("")
    report.append("")
    
    # Top 20
    report.append("TOP 20 RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")
    
    for idx, row in results_df.head(20).iterrows():
        report.append(f"#{idx + 1}: {row['title']}")
        report.append(f"   Author: {row.get('author', 'N/A')}")
        report.append(f"   Overall Score: {row['likelihood_score']:.1f}/100")
        report.append(f"   - Subject Similarity: {row['similarity_score']:.1f}")
        report.append(f"   - Checkout Volume: {row['checkout_volume_score']:.1f}")
        report.append(f"   - Author Popularity: {row['author_popularity_score']:.1f}")
        report.append("")
    
    return "\n".join(report)

# File upload section
st.header("📁 Step 1: Upload Your Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Checkouts File")
    st.markdown("""
    **Required columns:**
    - `title` - Book titles
    - `author` - Author names
    - `checkouts` - Number of checkouts
    - `lc_classification` (optional)
    - `subjects` (optional)
    """)
    checkouts_file = st.file_uploader("Upload checkouts CSV", type=['csv'], key='checkouts')

with col2:
    st.subheader("Recommendations File")
    st.markdown("""
    **Required columns:**
    - `title` - Book titles
    - `author` (optional)
    - `lc_classification` (optional)
    - `subjects` (optional)
    """)
    recommendations_file = st.file_uploader("Upload recommendations CSV", type=['csv'], key='recommendations')

# Process files if both are uploaded
if checkouts_file and recommendations_file:
    try:
        # Load data
        with st.spinner("Loading data..."):
            checkouts_df = pd.read_csv(checkouts_file)
            recommendations_df = pd.read_csv(recommendations_file)
        
        st.success(f"✅ Loaded {len(checkouts_df)} checkout records and {len(recommendations_df)} recommendations")
        
        # Show data preview
        with st.expander("📋 Preview Data"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Checkouts Data:**")
                st.dataframe(checkouts_df.head())
            with col2:
                st.write("**Recommendations Data:**")
                st.dataframe(recommendations_df.head())
        
        # Scoring configuration
        st.header("⚙️ Step 2: Configure Scoring Weights")
        st.markdown("Adjust how much each factor contributes to the final score:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            subject_weight = st.slider("Subject Similarity", 0.0, 1.0, 0.5, 0.05)
        with col2:
            lc_weight = st.slider("LC Classification", 0.0, 1.0, 0.3, 0.05)
        with col3:
            author_weight = st.slider("Author Popularity", 0.0, 1.0, 0.2, 0.05)
        
        total_weight = subject_weight + lc_weight + author_weight
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_weight:.2f}. They should sum to 1.0 for best results.")
        
        # Run analysis
        if st.button("🚀 Score Recommendations", type="primary"):
            with st.spinner("Analyzing recommendations... This may take a moment."):
                scorer = RecommendationScorer(checkouts_df)
                results_df = scorer.score_recommendations(
                    recommendations_df,
                    subject_weight=subject_weight,
                    lc_weight=lc_weight,
                    author_weight=author_weight
                )
            
            st.success("✅ Analysis complete!")
            
            # Store results in session state
            st.session_state['results'] = results_df
            
            # Display results
            st.header("📊 Step 3: Review Results")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Scored", len(results_df))
            with col2:
                high_priority = len(results_df[results_df['likelihood_score'] >= 70])
                st.metric("High Priority (70+)", high_priority)
            with col3:
                medium_priority = len(results_df[(results_df['likelihood_score'] >= 40) & 
                                                 (results_df['likelihood_score'] < 70)])
                st.metric("Medium Priority (40-69)", medium_priority)
            with col4:
                low_priority = len(results_df[results_df['likelihood_score'] < 40])
                st.metric("Low Priority (<40)", low_priority)
            
            # Filters
            st.subheader("🔍 Filter Results")
            col1, col2 = st.columns([2, 1])
            with col1:
                search = st.text_input("Search by title or author", "")
            with col2:
                min_score = st.slider("Minimum score", 0, 100, 0)
            
            # Apply filters
            filtered_df = results_df.copy()
            if search:
                mask = (filtered_df['title'].str.contains(search, case=False, na=False) | 
                       filtered_df['author'].str.contains(search, case=False, na=False))
                filtered_df = filtered_df[mask]
            filtered_df = filtered_df[filtered_df['likelihood_score'] >= min_score]
            
            # Display filtered results
            st.subheader(f"📚 Results ({len(filtered_df)} books)")
            
            # Format the dataframe for display
            display_df = filtered_df.copy()
            
            # Add priority labels
            def get_priority(score):
                if score >= 70:
                    return "🟢 High"
                elif score >= 40:
                    return "🟡 Medium"
                else:
                    return "🔴 Low"
            
            display_df['Priority'] = display_df['likelihood_score'].apply(get_priority)
            
            # Reorder columns
            priority_cols = ['Priority', 'title', 'author', 'likelihood_score', 
                           'similarity_score', 'checkout_volume_score', 'author_popularity_score']
            other_cols = [col for col in display_df.columns if col not in priority_cols]
            display_df = display_df[priority_cols + other_cols]
            
            # Show dataframe
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600
            )
            
            # Download section
            st.header("💾 Step 4: Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name="recommendations_with_scores.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Text report
                report = generate_report(results_df)
                st.download_button(
                    label="📄 Download Report (TXT)",
                    data=report,
                    file_name="recommendation_report.txt",
                    mime="text/plain"
                )
            
            # Top 20 preview
            st.subheader("🏆 Top 20 Recommendations")
            top_20 = results_df.head(20)[['title', 'author', 'likelihood_score', 
                                         'similarity_score', 'checkout_volume_score', 
                                         'author_popularity_score']]
            st.dataframe(top_20, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        st.info("""
        **Common issues:**
        - Check that your CSV files have the required columns
        - Ensure 'checkouts' column contains numeric values
        - Verify there are no special characters causing parsing issues
        """)

else:
    st.info("👆 Please upload both files to begin analysis")

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    ### Quick Start:
    1. Upload your **checkouts CSV**
    2. Upload your **recommendations CSV**
    3. Adjust scoring weights (optional)
    4. Click **Score Recommendations**
    5. Download your results!
    
    ### Interpreting Scores:
    
    **Likelihood Score (0-100):**
    - **70-100**: High priority
    - **40-69**: Medium priority  
    - **0-39**: Low priority
    
    **Component Scores:**
    - **Similarity**: Topic match
    - **Checkout Volume**: Similar books' popularity
    - **Author Popularity**: Author's track record
    
    ### Tips:
    - Include LC classification and subjects for better matching
    - Use at least 6-12 months of checkout data
    - Scores are relative to YOUR library's patterns
    """)
    
    st.divider()
    st.markdown("Made with ❤️ for librarians")
