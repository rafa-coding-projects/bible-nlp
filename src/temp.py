import numpy as np
import torch


# Load sentiment analysis model (lightweight and effective)
from transformers import pipeline
import requests
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import pandas as pd

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
verse_embeddings = np.load("verse_embeddings.npy")
df = pd.read_csv("/home/vdsuser/Projects/bible-nlp/bible_verses.csv")

# Initialize sentiment analyzer (runs once)
print("Loading sentiment analysis model...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)
print("✓ Sentiment model loaded")


def search_bible_cosine(query, top_k=10):
    query_vec = model.encode([query])[0]  # shape: (embedding_dim,)
    # Normalize embeddings
    verse_emb_norm = verse_embeddings / np.linalg.norm(verse_embeddings, axis=1, keepdims=True)
    query_vec_norm = query_vec / np.linalg.norm(query_vec)
    # Compute cosine similarities
    similarities = np.dot(verse_emb_norm, query_vec_norm)
    # Get top_k indices
    top_k_idx = np.argsort(similarities)[-top_k:][::-1]
    return df.iloc[top_k_idx][['verse_txt']].values


def score_verse_positivity(verse_text):
    """
    Score how positive/uplifting a verse is using a transformer-based sentiment model.
    Returns a score from 0 (very negative) to 1 (very positive).
    """
    # Truncate very long verses to fit model's max length
    text = verse_text[:512] if len(verse_text) > 512 else verse_text
    
    result = sentiment_analyzer(text)[0]
    
    # Convert to 0-1 scale where 1 is most positive
    if result['label'] == 'POSITIVE':
        return result['score']  # Already 0.5 to 1.0
    else:  # NEGATIVE
        return 1.0 - result['score']  # Convert to 0 to 0.5 range


def search_with_sentiment_filter(query, top_k=5, rerank_top_n=30, min_positivity=0.6):
    """
    Search with sentiment filtering to ensure uplifting, positive verses.
    Uses transformer-based sentiment analysis instead of keyword matching.
    """
    print(f"Query: {query}\n")
    
    # Get more candidates than needed
    initial_results = search_bible_cosine(query, top_k=rerank_top_n * 2)
    verses_list = [verse[0] for verse in initial_results]
    
    # Score verses by both relevance and positivity
    pairs = [[query, verse] for verse in verses_list]
    relevance_scores = cross_encoder.predict(pairs)
    
    # Use sentiment model for positivity scoring
    positivity_scores = np.array([score_verse_positivity(v) for v in verses_list])
    
    # Normalize and combine scores (70% relevance, 30% positivity)
    rel_norm = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min() + 1e-8)
    pos_norm = positivity_scores  # Already normalized 0-1
    
    combined_scores = 0.55 * rel_norm + 0.45 * pos_norm
    
    positive_mask = positivity_scores >= min_positivity
    filtered_indices = np.where(positive_mask)[0]
    
    if len(filtered_indices) < top_k:
        # If not enough positive verses, just use top combined scores
        top_idx = np.argsort(combined_scores)[-top_k:][::-1]
    else:
        # Get top_k from positive verses only
        filtered_scores = combined_scores[filtered_indices]
        top_in_filtered = np.argsort(filtered_scores)[-top_k:][::-1]
        top_idx = filtered_indices[top_in_filtered]
    
    return np.array(verses_list)[top_idx]


def search_with_diversity(query, top_k=5, rerank_top_n=30, diversity_penalty=0.3):
    """
    Enhanced search with MMR (Maximal Marginal Relevance) for diverse results.
    Avoids returning very similar verses.
    """
    print(f"Query: {query}\n")
    
    # Get more candidates
    initial_results = search_bible_cosine(query, top_k=rerank_top_n * 2)
    verses_list = [verse[0] for verse in initial_results]
    
    # Score by relevance and positivity
    pairs = [[query, verse] for verse in verses_list]
    relevance_scores = cross_encoder.predict(pairs)
    # positivity_scores = np.array([score_verse_positivity(v) for v in verses_list])
    
    # Normalize scores
    rel_norm = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min() + 1e-8)
    
    # Compute verse-to-verse similarity for diversity
    verse_embs = model.encode(verses_list)
    verse_embs_norm = verse_embs / np.linalg.norm(verse_embs, axis=1, keepdims=True)
    
    # MMR: Select diverse verses
    selected_indices = []
    selected_embeddings = []
    
    # Start with highest scoring verse
    candidates = np.arange(len(verses_list))
    # first_idx = np.argmax(rel_norm * positivity_scores)
    # selected_indices.append(first_idx)
    first_idx = np.argmax(relevance_scores)
    selected_indices.append(first_idx)
    selected_embeddings.append(verse_embs_norm[first_idx])
    candidates = np.delete(candidates, np.where(candidates == first_idx))
    
    # Iteratively select diverse verses
    while len(selected_indices) < top_k and len(candidates) > 0:
        # Calculate relevance and diversity scores
        # relevance = rel_norm[candidates] * positivity_scores[candidates]
        relevance = rel_norm[candidates]

        # Calculate similarity to already selected verses
        if len(selected_embeddings) > 0:
            selected_matrix = np.array(selected_embeddings)
            candidate_embs = verse_embs_norm[candidates]
            # Max similarity to any selected verse
            similarities = np.max(candidate_embs @ selected_matrix.T, axis=1)
        else:
            similarities = np.zeros(len(candidates))
        
        # MMR score: balance relevance and diversity
        mmr_scores = relevance - diversity_penalty * similarities
        
        # Select best MMR score
        best_candidate_idx = np.argmax(mmr_scores)
        best_idx = candidates[best_candidate_idx]
        
        selected_indices.append(best_idx)
        selected_embeddings.append(verse_embs_norm[best_idx])
        candidates = np.delete(candidates, best_candidate_idx)
    
    # return np.array(verses_list)[selected_indices]
    return [verses_list[idx] for idx in selected_indices]


def reformulate_query_with_llm(query, base_url="http://localhost:11434", model="llama3.2:3b"):
    """
    Use LLM to intelligently reformulate problem-focused queries into 
    solution/wisdom-focused queries for better Bible verse retrieval.
    """
    prompt = f"""Transform this problem-focused question into a positive, solution-focused search query for finding helpful Bible verses.

Original question: "{query}"

Rules:
- Focus on solutions, wisdom, and positive outcomes rather than problems
- Keep it concise (under 15 words)
- Use words like: wisdom, guidance, peace, comfort, strength, hope, faith, love
- Remove negative framing

Examples:
- "dealing with difficult people at work" → "wisdom for peaceful relationships at work"
- "I'm afraid of failure" → "courage and faith in challenging times"
- "feeling lonely and sad" → "God's comfort and companionship"

Reformulated query:"""

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused output
                    "max_tokens": 50,
                }
            },
            timeout=10
        )
        
        if response.status_code == 200:
            reformulated = response.json()['response'].strip().strip('"').strip()
            return reformulated
        else:
            # Fallback to simple heuristic
            return f"biblical wisdom and guidance for {query}"
            
    except:
        # Fallback if LLM not available
        return f"biblical wisdom and guidance for {query}"