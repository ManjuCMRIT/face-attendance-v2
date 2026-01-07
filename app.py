import streamlit as st
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

from firebase_utils import db
from face_processor import load_model
from face_matcher import find_best_match


# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
st.set_page_config("Face Attendance V2", layout="wide")
st.title("🧑‍🏫 Face Attendance System (V2)")

# --------------------------------------------------
# Load Face Model
# --------------------------------------------------
@st.cache_resource
def load_face_model():
    return load_model()

model = load_face_model()

# --------------------------------------------------
# CLASS SELECTION
# --------------------------------------------------
st.subheader("1️⃣ Select Class")

dept = st.selectbox("Department", ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"])
batch = st.text_input("Batch", "2024")
section = st.text_input("Section", "A")

if not (dept and batch and section):
    st.stop()

class_id = f"{dept}_{batch}_{section}"

docs = db.collection("classes").document(class_id).collection("students").stream()
students = {s.id: s.to_dict() for s in docs}

if not students:
    st.error("⚠ No student list found. Upload students via Admin Dashboard.")
    st.stop()

st.success(f"Loaded **{len(students)} students** for `{class_id}`")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "present" not in st.session_state:
    st.session_state.present = set()

if "manual_present" not in st.session_state:
    st.session_state.manual_present = set()

if "unknown_faces" not in st.session_state:
    st.session_state.unknown_faces = []

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
st.subheader("2️⃣ Upload Classroom Image")

file = st.file_uploader("Upload classroom image", type=["jpg", "png"])

if file and st.button("📸 Process Image"):
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)

    faces = model.get(img_np)

    if not faces:
        st.warning("No faces detected.")
        st.stop()

    for face in faces:
        emb = face.embedding
        usn, score = find_best_match(emb, students)

        x1, y1, x2, y2 = face.bbox.astype(int)

        if usn != "Unknown":
            st.session_state.present.add(usn)
            label = f"{students[usn]['name']} ({usn})"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)
            st.session_state.unknown_faces.append(img_np[y1:y2, x1:x2])

        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_np, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    st.image(img_np, use_column_width=True)
    st.info("👉 Upload another image if needed. Attendance accumulates.")

# --------------------------------------------------
# ATTENDANCE SUMMARY
# --------------------------------------------------
st.subheader("📊 Attendance Summary")

auto_present = set(st.session_state.present)
absent = [u for u in students if u not in auto_present]

st.write(f"🟢 Automatically Present: **{len(auto_present)}**")
for usn in auto_present:
    st.write(f"✔ {students[usn]['name']} ({usn})")

# --------------------------------------------------
# MANUAL OVERRIDE (CHECKBOXES)
# --------------------------------------------------
st.subheader("🔴 Absent Students (Manual Override)")

st.caption("✔ Tick students who are present but not recognized")

for usn in absent:
    label = f"{students[usn]['name']} ({usn})"

    checked = st.checkbox(
        label,
        key=f"manual_{usn}",
        value=(usn in st.session_state.manual_present)
    )

    if checked:
        st.session_state.manual_present.add(usn)
    else:
        st.session_state.manual_present.discard(usn)

# --------------------------------------------------
# UNKNOWN FACES
# --------------------------------------------------
if st.session_state.unknown_faces:
    st.subheader("🟡 Unknown Detected Faces")
    cols = st.columns(5)
    for i, face_img in enumerate(st.session_state.unknown_faces):
        with cols[i % 5]:
            st.image(face_img)

# --------------------------------------------------
# FINALIZE ATTENDANCE
# --------------------------------------------------
st.divider()

if st.button("✅ Finalize Attendance"):
    final_present = auto_present | st.session_state.manual_present
    final_absent = [u for u in students if u not in final_present]

    today = datetime.now().strftime("%Y-%m-%d")

    db.collection("classes").document(class_id) \
      .collection("attendance").document(today).set({
          "present": list(final_present),
          "absent": final_absent,
          "unknown_count": len(st.session_state.unknown_faces),
          "timestamp": datetime.now()
      })

    st.success(f"Attendance saved for **{class_id} on {today}** 🎉")
    st.balloons()

    # Reset session
    st.session_state.present.clear()
    st.session_state.manual_present.clear()
    st.session_state.unknown_faces.clear()
