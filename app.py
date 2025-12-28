import streamlit as st
import numpy as np
from PIL import Image
import cv2

from firebase_utils import db
from face_processor import load_model, get_embedding
from face_matcher import find_best_match

# --------------------------------------------------------------
st.set_page_config("Face Attendance V2", layout="wide")
st.title("🧑‍🏫 Face Attendance System - Version 2")

# Load model once
@st.cache_resource
def load_face_model():
    return load_model()

model = load_face_model()

# --------------------------------------------------------------
# 1. Select Class
st.subheader("1. Select Class")

dept = st.selectbox("Department", ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"])
batch = st.text_input("Batch", "2024")
section = st.text_input("Section", "A")

if not (dept and batch and section):
    st.stop()

class_id = f"{dept}_{batch}_{section}"

# Load class students
docs = db.collection("classes").document(class_id).collection("students").stream()
students = {s.id:s.to_dict() for s in docs}

if not students:
    st.error("⚠ No student list found. Upload via Admin Dashboard.")
    st.stop()

st.success(f"Loaded **{len(students)} students** for `{class_id}`")

# --------------------------------------------------------------
# Session variables to accumulate attendance
if "present" not in st.session_state: st.session_state.present = set()
if "unknown_list" not in st.session_state: st.session_state.unknown_list = []
if "processed_count" not in st.session_state: st.session_state.processed_count = 0

# --------------------------------------------------------------
# 2. Upload Image
st.subheader("2. Upload Classroom Photo")

file = st.file_uploader("Upload image", type=["jpg","png"])

if file and st.button("Process Image"):
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)

    faces = model.get(img_np)
    st.session_state.processed_count += 1

    if len(faces) == 0:
        st.warning("No faces detected.")
        st.stop()

    for face in faces:
        emb = face.embedding
        usn, score = find_best_match(emb, students, threshold=0.55)   # <-- COSINE MATCHING

        x1,y1,x2,y2 = face.bbox.astype(int)

        if usn != "Unknown":
            st.session_state.present.add(usn)
            label = f"{students[usn]['name']} ({score:.2f})"
            color = (0,255,0)
        else:
            crop = img_np[y1:y2, x1:x2]
            st.session_state.unknown_list.append(crop)
            label = f"Unknown ({score:.2f})"
            color = (0,0,255)

        cv2.rectangle(img_np,(x1,y1),(x2,y2),color,2)
        cv2.putText(img_np,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    st.image(img_np, caption=f"Processed Image #{st.session_state.processed_count}", use_column_width=True)

    # Attendance calculation
    absent = [usn for usn in students if usn not in st.session_state.present]

    # ---------------- Display Results ----------------
    st.subheader("🟢 Present Students")
    for usn in st.session_state.present:
        st.write(f"✔ {students[usn]['name']} ({usn})")

    st.subheader("🔴 Absent Students")
    if absent:
        st.write(absent)
    else:
        st.success("Everyone marked present!")

    st.subheader("🟡 Unknown Faces")
    cols = st.columns(4)
    for i, face_crop in enumerate(st.session_state.unknown_list):
        with cols[i % 4]:
            st.image(face_crop)

    st.info("👉 Upload another image to improve detection. Attendance accumulates.")


# --------------------------------------------------------------
# 3. Finalize Attendance (next feature to expand)
if st.button("Finalize Attendance"):
    st.success("📌 Attendance Finalized — (Next Step will store to DB)")
    st.write("Present:", list(st.session_state.present))
    st.write("Unknown faces:", len(st.session_state.unknown_list))

    # In next iteration — store to Firestore with date/time/class etc.


