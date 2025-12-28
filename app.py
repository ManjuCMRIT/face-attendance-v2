import streamlit as st
import numpy as np
from PIL import Image
import cv2
from firebase_utils import db
from face_processor import load_model, get_embedding
from face_matcher import find_best_match

# UI
st.set_page_config("Face Attendance V2", layout="wide")
st.title("🧑‍🏫 Face Attendance V2 - (Multi Image Ready)")

# Load model cached
@st.cache_resource
def load_face_model():
    return load_model()

model = load_face_model()

# --------------- CLASS SELECTION ----------------
st.subheader("1. Select Class")
dept = st.selectbox("Department", ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"])
batch = st.text_input("Batch", "2024")
section = st.text_input("Section", "A")

if not (dept and batch and section):
    st.stop()

class_id = f"{dept}_{batch}_{section}"

docs = db.collection("classes").document(class_id).collection("students").stream()
students = {s.id:s.to_dict() for s in docs}

if not students:
    st.error("No student list found. Upload in Admin Dashboard.")
    st.stop()

st.success(f"Loaded {len(students)} students for {class_id}")

# State for multi-image accumulation
if "present" not in st.session_state: st.session_state.present = set()
if "unknown_list" not in st.session_state: st.session_state.unknown_list = []


# --------------- IMAGE UPLOAD ----------------
st.subheader("2. Upload Classroom Photo")
file = st.file_uploader("Upload image", type=["jpg","png"])


if file and st.button("Process Image"):
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)

    faces = model.get(img_np)

    if len(faces) == 0:
        st.error("No faces detected in image")
        st.stop()

    for face in faces:
        emb = face.embedding
        usn, score = find_best_match(emb, students)

        x1,y1,x2,y2 = face.bbox.astype(int)

        if usn != "Unknown":
            st.session_state.present.add(usn)
            cv2.putText(img_np, students[usn]["name"], (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        else:
            crop = img_np[y1:y2, x1:x2]
            st.session_state.unknown_list.append(crop)
            cv2.putText(img_np, "Unknown", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        cv2.rectangle(img_np,(x1,y1),(x2,y2),(0,255,0),2)

    st.image(img_np, use_column_width=True)

    absent = [s for s in students if s not in st.session_state.present]

    st.subheader("🟢 Present Students")
    for usn in st.session_state.present:
        st.write(f"{students[usn]['name']} ({usn})")

    st.subheader("🔴 Absent Students")
    st.write(absent)

    st.subheader("🟡 Unknown Faces")
    cols = st.columns(4)
    i=0
    for f in st.session_state.unknown_list:
        with cols[i%4]:
            st.image(f)
        i+=1

    st.warning("Upload another image if more faces remain → attendance accumulates")


# --------------- FINALIZE ATTENDANCE (Phase-2 will expand) ----------------
if st.button("Finalize Attendance"):
    st.success("Attendance finalization step will be added next 🚀")
    st.write("Present:", st.session_state.present)
    st.write("Unknown count:", len(st.session_state.unknown_list))
