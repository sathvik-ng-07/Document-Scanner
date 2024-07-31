import streamlit as st
import cv2
import numpy as np
import pytesseract

# Update this path to where Tesseract is installed on your machine
pytesseract.pytesseract.tesseract_cmd = r'F:\program_files\teseract\tesseract.exe'

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return gray, edges

def detect_document(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx
    return None

def apply_perspective_transform(image, document_contour):
    pts = document_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def extract_text_from_image(warped):
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(warped_gray)
    return text

# Streamlit UI
st.title("Document Scanner")

uploaded_file = st.file_uploader("Choose a PNG file", type=["png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    gray, edges = preprocess_image(image)
    document_contour = detect_document(edges)
    
    if document_contour is not None:
        warped = apply_perspective_transform(image, document_contour)
        text = extract_text_from_image(warped)
        
        st.subheader("Scanned Document")
        st.image(warped, channels="BGR", use_column_width=True)
        
        st.subheader("Extracted Text")
        st.text(text)
    else:
        st.error("No document detected.")
