import streamlit as st
import numpy as np
from PIL import Image
import cv2
from firebase_utils import db
from face_processor import load_model
from face_matcher import find_best_match      # using wrapper


# ------------------- Streamlit UI -------------------
st.set_page_config("Face Attendance V2", layout="wide")
st.title("🧑‍🏫 Face Attendance V2 - Multi Image Attendance")


@st.cache_resource
def load_face_model():
    return load_model()

model = load_face_model()


# ------------------- 1. CLASS SELECTION -------------------
st.subheader("1. Select Class")

dept = st.selectbox("Department", ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"])
batch = st.text_input("Batch", "2024")
section = st.text_input("Section", "A")

if not (dept and batch and section):
    st.stop()

class_id = f"{dept}_{batch}_{section}"

docs = db.collection("classes").document(class_id).collection("students").stream()
students = {s.id: s.to_dict() for s in docs}

if not students:
    st.error("⚠ No student list found. Upload via Admin Dashboard first.")
    st.stop()

st.success(f"Loaded {len(students)} students for **{class_id}**")


# ------------------- INIT SESSION -------------------
if "present" not in st.session_state: st.session_state.present = set()
if "unknown_faces" not in st.session_state: st.session_state.unknown_faces = []


# ------------------- 2. UPLOAD IMAGE -------------------
st.subheader("2. Upload Classroom Photo")
image_file = st.file_uploader("Upload classroom image", type=["jpg","png"])

if image_file and st.button("Process Image"):
    img = Image.open(image_file).convert("RGB")
    img_np = np.array(img)

    faces = model.get(img_np)

    if len(faces) == 0:
        st.error("❌ No faces detected.")
        st.stop()

    # PREPARE REGISTERED EMBEDDINGS
    registered_embeddings = {u: s.get("embedding") for u,s in students.items() if "embedding" in s}

    for face in faces:
        emb = face.embedding                       # embedding from image
        x1, y1, x2, y2 = face.bbox.astype(int)

        # ---------- Debug: Check embedding size ----------
        st.write("Detected embedding length =", len(emb))
        if len(registered_embeddings) > 0:
            sample_user = next(iter(registered_embeddings.values()))
            if sample_user:
                st.write("Stored embedding length =", len(sample_user))
        # --------------------------------------------------

        usn, score = find_best_match(emb, registered_embeddings)

        # ================= MATCH FOUND ==================
        if usn:
            name = students[usn]['name']
            st.session_state.present.add(usn)

            cv2.rectangle(img_np,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img_np,f"{name} ({usn})",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        # ================= UNKNOWN ==================
        else:
            face_crop = img_np[y1:y2, x1:x2]
            st.session_state.unknown_faces.append(face_crop)

            cv2.rectangle(img_np,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(img_np,"Unknown",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    st.image(img_np, use_column_width=True)

    # Calculate absent
    absent = [usn for usn in students if usn not in st.session_state.present]


    # =================== DISPLAY RESULTS ===================
    st.subheader("🟢 Present Students")
    for usn in st.session_state.present:
        st.write(f"✔ {students[usn]['name']} ({usn})")

    st.subheader("🔴 Absent Students")
    st.write(absent)

    st.subheader("🟡 Unknown Faces")
    cols = st.columns(4)
    for i, face in enumerate(st.session_state.unknown_faces):
        with cols[i % 4]:
            st.image(face)

    st.info("📌 Upload more images if needed — attendance will merge automatically.")


# ------------------- 3. FINALIZE ATTENDANCE -------------------
if st.button("Finalize Attendance"):
    st.success("Finalization page coming next 🚀")
    st.write("Present:", list(st.session_state.present))
    st.write("Unknown:", len(st.session_state.unknown_faces))
