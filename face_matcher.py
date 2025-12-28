import numpy as np
import cv2
import insightface

# Load recognition model once
_model = None
_face_app = None

def load_model():
    global _model, _face_app
    if _model is None:
        _face_app = insightface.app.FaceAnalysis(name="buffalo_l")
        _face_app.prepare(ctx_id=0)
    return _face_app


# Extract embeddings for multiple faces in classroom photo
def get_embeddings(img_np):
    app = load_model()
    faces = app.get(img_np)

    result = []
    for f in faces:
        emb = f.embedding.tolist()
        bbox = list(map(int, f.bbox))  # (x1,y1,x2,y2)
        result.append((emb, bbox))

    return result


# Compare embeddings with registered students
def match_faces(emb, registered, threshold=0.55):
    """
    emb = embedding from input image
    registered = {usn:[vector], ...}
    returns → (matched_usn, distance_score)
    """

    emb = np.array(emb)

    best_match = None
    best_score = 99

    for usn, reg_emb in registered.items():
        reg_emb = np.array(reg_emb)
        dist = np.linalg.norm(reg_emb - emb)

        if dist < best_score and dist < threshold:
            best_score = dist
            best_match = usn

    return best_match, best_score

# ----------------------------------------
# Backward compatibility for old code
# ----------------------------------------
def find_best_match(embedding, registered_users, threshold=0.55):
    """Wrapper for older apps that used find_best_match"""
    return match_faces(embedding, registered_users, threshold)

