import streamlit as st
import pandas as pd
from firebase_utils import db

st.set_page_config(
    page_title="Attendance Report",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Attendance Report (Excel)")
st.caption("Class-wise attendance report for all students")

st.divider()

# --------------------------------------------------
# CLASS SELECTION
# --------------------------------------------------
st.subheader("Select Class")

dept = st.selectbox(
    "Department",
    ["CSE","ISE","AI/ML","CS-ML","CS-DS","AI/DS","MBA","MCA"]
)
batch = st.text_input("Batch", "2024")
section = st.text_input("Section", "A")

if not (dept and batch and section):
    st.stop()

class_id = f"{dept}_{batch}_{section}"

# --------------------------------------------------
# LOAD STUDENTS
# --------------------------------------------------
students_ref = db.collection("classes").document(class_id)\
                 .collection("students").stream()

students = {s.id: s.to_dict() for s in students_ref}

if not students:
    st.error("No students found for this class.")
    st.stop()

# --------------------------------------------------
# LOAD ATTENDANCE RECORDS
# --------------------------------------------------
attendance_ref = db.collection("classes").document(class_id)\
                   .collection("attendance").stream()

attendance_docs = list(attendance_ref)

if not attendance_docs:
    st.warning("No attendance records found for this class.")
    st.stop()

# --------------------------------------------------
# BUILD DATAFRAME
# --------------------------------------------------
# Base table
df = pd.DataFrame(
    [{"USN": usn, "Name": data["name"]} for usn, data in students.items()]
)

# Sort attendance by date
attendance_docs.sort(key=lambda d: d.id)

# Add each date as a column
for doc in attendance_docs:
    data = doc.to_dict()
    date = doc.id
    present_list = data.get("present", [])

    df[date] = df["USN"].apply(
        lambda u: "P" if u in present_list else "AB"
    )

st.subheader("Attendance Table")
st.dataframe(df, use_container_width=True)

# --------------------------------------------------
# DOWNLOAD EXCEL
# --------------------------------------------------
st.divider()

if st.button("📥 Download Attendance Excel"):
    file_name = f"{class_id}_attendance_report.xlsx"
    df.to_excel(file_name, index=False)

    with open(file_name, "rb") as f:
        st.download_button(
            label="Click here to download",
            data=f,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
