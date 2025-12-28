import streamlit as st
import numpy as np
from PIL import Image
import cv2
from firebase_utils import db
from face_processor import load_model, get_embedding
from face_matcher import find_best_match

# ------------------- APP UI -------------------
st.set_page_config("Face Attendance V2", layout="wide")
st.title("🧑‍🏫 Face Attendance System (V2 - Multi Image Support)")

# ------------------ Load Face Model ------------------
@st.cache_resource
def load_face_model():
    return load_model()

model = load_face_model()


# ------------------ CLASS SELECTION ------------------
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
    st.error("⚠ No student list found. Upload using Admin Dashboard.")
    st.stop()

st.success(f"Loaded {len(students)} students for class **{class_id}**")


# ------------------ Session Memory ------------------
if "present" not in st.session_state: st.session_state.present = set()
if "unknown_list" not in st.session_state: st.session_state.unknown_list = []

st.divider()


# ------------------ IMAGE UPLOAD ------------------
st.subheader("2. Upload Classroom Image")

file = st.file_uploader("Choose an image and click *Process Image*", type=["jpg","png"], key="img1")

if file and st.button("📸 Process Image"):
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)

    faces = model.get(img_np)

    if len(faces) == 0:
        st.error("No faces detected — try a clearer image.")
        st.stop()

    for face in faces:
        emb = face.embedding
        usn, score = find_best_match(emb, students)

        x1,y1,x2,y2 = face.bbox.astype(int)

        if usn != "Unknown":
            st.session_state.present.add(usn)
            label = f"{students[usn]['name']} ({usn})"
            color = (0,255,0)
        else:
            label = "Unknown"
            color = (0,0,255)
            st.session_state.unknown_list.append(img_np[y1:y2, x1:x2])

        cv2.rectangle(img_np,(x1,y1),(x2,y2),color,2)
        cv2.putText(img_np,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    st.image(img_np, use_column_width=True)

    st.info("👉 Upload another image below to improve detection. Attendance continues accumulating.")


# ------------------ UPLOAD NEXT IMAGE ------------------
st.subheader("Upload Another Angle")

next_img = st.file_uploader("Upload next image", type=["jpg","png"], key="img2")

if next_img and st.button("➡ Process Next Image"):
    st.session_state.img1 = next_img
    st.rerun()


# ------------------ DISPLAY ATTENDANCE ------------------
st.subheader("📊 Attendance Summary")

present = list(st.session_state.present)
absent = [usn for usn in students if usn not in present]

st.write(f"🟢 Present: **{len(present)}**")
for usn in present:
    st.write(f"- {students[usn]['name']} ({usn})")

st.write(f"\n🔴 Absent: **{len(absent)}**")
for usn in absent:
    st.write(f"- {students[usn]['name']} ({usn})")


# ------------------ Unknown Faces ------------------
if st.session_state.unknown_list:
    st.subheader("🟡 Unknown Detected Faces")
    cols = st.columns(5)
    i = 0
    for face_img in st.session_state.unknown_list:
        with cols[i % 5]:
            st.image(face_img)
        i += 1


# ------------------ Finalize Attendance ------------------
if st.button("Finalize Attendance"):
    st.success("Attendance finalization backend coming next... 📝 (Save to Firebase)")

    st.write("✔ Present Students:", present)
    st.write("❌ Absent Students:", absent)
    st.write("🟡 Unknown count:", len(st.session_state.unknown_list))

    # Here we will save to database --> next task
