import streamlit as st
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

from firebase_utils import db
from face_processor import load_model
from face_matcher import find_best_match


# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🧑‍🏫",
    layout="wide"
)

st.title("🧑‍🏫 Face Attendance System")
st.caption("AI-assisted attendance with manual verification")

st.divider()


# ==================================================
# LOAD FACE MODEL
# ==================================================
@st.cache_resource
def load_face_model():
    return load_model()

model = load_face_model()


# ==================================================
# STEP 1: CLASS SELECTION
# ==================================================
st.subheader("1️⃣ Select Class")

dept = st.selectbox(
    "Department",
    ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"]
)
batch = st.text_input("Batch (Year of Admission)", "2024")
section = st.text_input("Section", "A")

if not (dept and batch and section):
    st.info("Please select class details to proceed.")
    st.stop()

class_id = f"{dept}_{batch}_{section}"

docs = db.collection("classes").document(class_id)\
          .collection("students").stream()

students = {s.id: s.to_dict() for s in docs}

if not students:
    st.error("No students found. Upload student list via Admin Dashboard.")
    st.stop()

st.success(f"Class loaded: **{class_id}** | Students: **{len(students)}**")

st.divider()


# ==================================================
# SESSION STATE
# ==================================================
if "present" not in st.session_state:
    st.session_state.present = set()

if "manual_present" not in st.session_state:
    st.session_state.manual_present = set()

if "unknown_faces" not in st.session_state:
    st.session_state.unknown_faces = []

if "images_processed" not in st.session_state:
    st.session_state.images_processed = 0


# ==================================================
# STEP 2: IMAGE UPLOAD
# ==================================================
st.subheader("2️⃣ Upload Classroom Image")

st.info(
    "Upload one or more classroom images from different angles. "
    "Attendance accumulates across uploads."
)

file = st.file_uploader(
    "Upload classroom image (jpg / png)",
    type=["jpg", "png"]
)

if file and st.button("📸 Process Image"):
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)

    faces = model.get(img_np)
    st.session_state.images_processed += 1

    if not faces:
        st.warning("No faces detected in this image.")
    else:
        for face in faces:
            emb = face.embedding
            usn, score = find_best_match(emb, students)

            x1, y1, x2, y2 = face.bbox.astype(int)

            if usn != "Unknown":
                st.session_state.present.add(usn)
                label = f"{students[usn]['name']} ({usn})"
                color = (0, 180, 0)
            else:
                label = "Unknown"
                color = (200, 0, 0)
                st.session_state.unknown_faces.append(
                    img_np[y1:y2, x1:x2]
                )

            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img_np, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        st.image(
            img_np,
            caption=f"Processed Image #{st.session_state.images_processed}",
            use_column_width=True
        )

st.divider()


# ==================================================
# STEP 3: ATTENDANCE SUMMARY
# ==================================================
st.subheader("3️⃣ Attendance Summary")

auto_present = set(st.session_state.present)
absent = [u for u in students if u not in auto_present]

col1, col2, col3 = st.columns(3)
col1.metric("Images Processed", st.session_state.images_processed)
col2.metric("Automatically Present", len(auto_present))
col3.metric("Total Students", len(students))


# ==================================================
# STEP 4: MANUAL OVERRIDE
# ==================================================
st.subheader("4️⃣ Manual Verification (If Required)")

st.caption(
    "Tick students who are present but were not recognized by the system."
)

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


# ==================================================
# UNKNOWN FACES
# ==================================================
if st.session_state.unknown_faces:
    st.subheader("🟡 Unrecognized Faces")
    st.caption("Faces detected but not matched to any student")

    cols = st.columns(5)
    for i, face_img in enumerate(st.session_state.unknown_faces):
        with cols[i % 5]:
            st.image(face_img)


st.divider()


# ==================================================
# STEP 5: FINALIZE ATTENDANCE
# ==================================================
st.subheader("5️⃣ Finalize Attendance")

final_present = auto_present | st.session_state.manual_present
final_absent = [u for u in students if u not in final_present]

colA, colB, colC = st.columns(3)
colA.metric("Final Present", len(final_present))
colB.metric("Final Absent", len(final_absent))
colC.metric("Unknown Faces", len(st.session_state.unknown_faces))


if st.button("✅ Confirm & Save Attendance"):
    today = datetime.now().strftime("%Y-%m-%d")

    db.collection("classes").document(class_id) \
      .collection("attendance").document(today).set({
          "present": list(final_present),
          "absent": final_absent,
          "unknown_count": len(st.session_state.unknown_faces),
          "timestamp": datetime.now()
      })

    # ---------------- CONFIRMATION PANEL ----------------
    st.markdown("## ✅ Attendance Recorded Successfully")

    c1, c2, c3 = st.columns(3)
    c1.metric("Class", class_id)
    c2.metric("Date", today)
    c3.metric("Students", len(students))

    c4, c5, c6 = st.columns(3)
    c4.metric("Present", len(final_present))
    c5.metric("Absent", len(final_absent))
    c6.metric("Images Used", st.session_state.images_processed)

    st.success(
        "Attendance has been securely saved. "
        "You may start a new session or proceed to reports."
    )

    if st.button("🔄 Start New Attendance Session"):
        st.session_state.present.clear()
        st.session_state.manual_present.clear()
        st.session_state.unknown_faces.clear()
        st.session_state.images_processed = 0
        st.experimental_rerun()
