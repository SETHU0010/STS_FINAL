import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import logging
import re

# Set up logging for error tracking
logging.basicConfig(level=logging.ERROR)

# Threshold constants for similarity categories
MATCHED_THRESHOLD = 80
REVIEW_THRESHOLD = 50

# Keywords to check for critical differences
CRITICAL_KEYWORDS = ["Seller", "Buyer"]

# Function to check if critical keywords differ between two sentences
def check_critical_difference(sentence1, sentence2):
    keywords1 = set(re.findall(r'\b\w+\b', sentence1)) & set(CRITICAL_KEYWORDS)
    keywords2 = set(re.findall(r'\b\w+\b', sentence2)) & set(CRITICAL_KEYWORDS)
    return keywords1 != keywords2

# Function to categorize semantic deviation based on similarity percentage and critical differences
def categorize_semantic_deviation(similarity_percentage, sentence1, sentence2):
    if check_critical_difference(sentence1, sentence2):
        # Lower threshold if critical keywords differ
        if similarity_percentage >= MATCHED_THRESHOLD + 5:
            return "Matched with Critical Difference"
        elif similarity_percentage >= REVIEW_THRESHOLD:
            return "Need Review - Critical Difference"
        else:
            return "Missed Clause - Critical Difference"
    else:
        # Standard categorization if no critical keywords differ
        if similarity_percentage >= MATCHED_THRESHOLD:
            return "Matched"
        elif similarity_percentage >= REVIEW_THRESHOLD:
            return "Need Review"
        else:
            return "Missed Clause"

# Function to create a downloadable Excel file
def create_download_link(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Similarity Results')
    output.seek(0)
    return output

# Batch processing for large datasets
def batch_process(df, batch_size=100):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

# Function to calculate similarity using TF-IDF and cosine similarity
def calculate_similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_percentage = similarity_score[0][0] * 100
    return similarity_score[0][0], similarity_percentage

# Main function to define app layout
def main():
    st.title("Semantic Text Similarity (STS) Test")

    # Use Tabs for smoother navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "About this App", "📂 Upload Data", "✍️ Manual Input"])

    # Home Page - Tab 1
    with tab1:
        st.markdown("""
        <h2 style='font-size:28px;'>Welcome to the Semantic Similarity App</h2>
        <p style='font-size:16px;'>This app calculates the semantic similarity between pairs of sentences using TF-IDF and cosine similarity. The similarity score ranges from 0% (completely different) to 100% (identical meaning).</p>
        <p style='font-size:16px;'>Based on the similarity percentage, sentences are categorized into the following:</p>
        <ul style='font-size:16px;'>
            <li><strong>Matched:</strong> Similarity of 80% or above.</li>
            <li><strong>Need Review:</strong> Similarity between 50% and 79%.</li>
            <li><strong>Missed Clause:</strong> Similarity below 50%.</li>
            <li><strong>Matched with Critical Difference:</strong> High similarity but with differing critical keywords.</li>
        </ul>
        <p style='font-size:16px;'>To get started, choose <strong>Upload Data</strong> to process an Excel file or <strong>Manual Input</strong> to input sentences directly.</p>
        """, unsafe_allow_html=True)

    # About this App - Tab 2
    with tab2:
        st.markdown("---")
        st.write("### About this App")
        st.write("""
        This app uses TF-IDF and cosine similarity to calculate the semantic similarity between two sentences. It offers the following features:
        - 📊 **Batch Processing:** Efficiently handles large datasets for similarity calculations.
        - ⚙️ **TF-IDF & Cosine Similarity:** Utilizes advanced text processing techniques for accurate results.
        - 📥 **Download Results:** Provides options to download similarity results in Excel format.
        - 🔍 **Manual Input:** Allows users to input sentences directly for quick analysis.
        - 🔑 **Critical Keyword Check:** Identifies semantic differences based on critical keywords.
        """)

    # Upload Data - Tab 3
    with tab3:
        uploaded_file = st.file_uploader("Upload an Excel file with two columns for sentence pairs", type=["xlsx"])

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                if df.shape[1] < 2:
                    st.error("The uploaded file must contain at least two columns.")
                    return

                st.write("Uploaded Data:")
                st.dataframe(df)

                sentence1_col = df.columns[0]
                sentence2_col = df.columns[1]

                if st.button("Calculate Similarity", key="upload_button"):
                    results = []
                    progress_bar = st.progress(0)

                    # Process rows in batches
                    total_batches = len(df) // 100 + (len(df) % 100 > 0)
                    for i, batch in enumerate(batch_process(df)):
                        for _, row in batch.iterrows():
                            sentence1 = row[sentence1_col]
                            sentence2 = row[sentence2_col]

                            if pd.isna(sentence1) or pd.isna(sentence2):
                                similarity_percentage = 0
                            else:
                                _, similarity_percentage = calculate_similarity(sentence1, sentence2)

                            deviation_category = categorize_semantic_deviation(similarity_percentage, sentence1, sentence2)
                            results.append({
                                "Sentence 1": sentence1,
                                "Sentence 2": sentence2,
                                "Similarity Percentage": round(similarity_percentage, 2),
                                "Semantic Deviation": deviation_category
                            })

                        progress_bar.progress(min((i + 1) / total_batches, 1.0))

                    results_df = pd.DataFrame(results)
                    st.write("Similarity Results:")
                    st.dataframe(results_df)

                    excel_data = create_download_link(results_df)
                    st.download_button(
                        label="Download Results as Excel",
                        data=excel_data,
                        file_name="similarity_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except Exception as e:
                logging.error(f"Error processing file: {e}")
                st.error(f"Error processing file: {e}")

    # Manual Input - Tab 4
    with tab4:
        sentence1 = st.text_area("Enter the first sentence:")
        sentence2 = st.text_area("Enter the second sentence:")

        if st.button("Calculate Similarity", key="manual_button"):
            if sentence1 and sentence2:
                try:
                    similarity_score, similarity_percentage = calculate_similarity(sentence1, sentence2)
                    deviation_category = categorize_semantic_deviation(similarity_percentage, sentence1, sentence2)

                    st.write(f"**Similarity Score:** {similarity_score:.4f}")
                    st.write(f"**Similarity Percentage:** {similarity_percentage:.2f}%")
                    st.write(f"**Semantic Deviation:** {deviation_category}")

                    result_data = [{
                        "Sentence 1": sentence1,
                        "Sentence 2": sentence2,
                        "Similarity Score": round(similarity_score, 4),
                        "Similarity Percentage": round(similarity_percentage, 2),
                        "Semantic Deviation": deviation_category
                    }]
                    results_df = pd.DataFrame(result_data)

                    excel_data = create_download_link(results_df)
                    st.download_button(
                        label="Download Result as Excel",
                        data=excel_data,
                        file_name="manual_similarity_result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as e:
                    logging.error(f"Error calculating similarity: {e}")
                    st.error(f"Error calculating similarity: {e}")
            else:
                st.warning("Please enter both sentences for similarity calculation.")

if __name__ == "__main__":
    main()
