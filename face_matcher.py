import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def normalize(v):
    """Normalize vectors for better similarity score stability."""
    v = np.array(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-10)

def find_best_match(face_embedding, students, threshold=0.5):
    """
    face_embedding: numpy array (512,)
    students: dict {USN: {name, embedding}}
    Returns: usn, score
    """

    face_embedding = normalize(face_embedding).reshape(1, -1)

    best_usn = "Unknown"
    best_score = -1

    for usn, data in students.items():

        if "embedding" not in data:
            continue  # skip students without face registered

        emb = normalize(data["embedding"]).reshape(1, -1)
        score = cosine_similarity(face_embedding, emb)[0][0]

        if score > best_score:
            best_score = score
            best_usn = usn

    # threshold tuning
    if best_score >= threshold:
        return best_usn, float(best_score)

    return "Unknown", float(best_score)
