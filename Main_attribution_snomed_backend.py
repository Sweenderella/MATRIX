import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util

def find_closest_match(terms, target_term, model):
    """Find the closest matching term using embeddings."""
    term_embeddings = model.encode(terms, convert_to_tensor=True)
    target_embedding = model.encode(target_term, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(target_embedding, term_embeddings)
    closest_idx = similarities.argmax().item()
    return closest_idx

def search_snomed(diagnosis, csv_file):
    # Load the SNOMED CT data
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")  # Try UTF-8 first
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding="latin1")  # Fallback to latin1 if UTF-8 fails


     # Preprocess diagnosis for better matching
    diagnosis_keywords = diagnosis.lower().split()  # Split diagnosis into individual keywords
    keyword_pattern = '|'.join(diagnosis_keywords)  # Create a regex pattern for keywords
     
    matching_terms = df[df['term'].str.lower().str.contains(keyword_pattern, na=False)]
        

    

    # Filter rows where the "term" column matches the diagnosis
    #matching_terms = df[df['term'].str.contains(diagnosis, case=False, na=False)]
    
    if matching_terms.empty:
        return {"error": "No matching term found for the diagnosis."}
    
    if len(matching_terms) == 1:
        # If there's only one match, return it directly
        result = matching_terms.iloc[0]
    else:
        # Use vector embeddings to find the closest match
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        closest_idx = find_closest_match(matching_terms['term'].tolist(), diagnosis, model)
        result = matching_terms.iloc[closest_idx]
    
    # Prepare the output
    output = {
        "moduleId": result['moduleId'],
        "conceptId": result['conceptId'],
        "languageCode": result['languageCode'],
        "typeId": result['typeId'],
        "snomed_concept":result['term'], 
        "caseSignificanceId": result['caseSignificanceId'],
        "message": f"The diagnosis made by AI refers to the clinical term specified by: "
                   f"moduleId {result['moduleId']}, conceptId {result['conceptId']}, "
                   f"languageCode {result['languageCode']}, typeId {result['typeId']}, "
                   f"caseSignificanceId {result['caseSignificanceId']}. This information can be found on the official website."
    }
    return output

