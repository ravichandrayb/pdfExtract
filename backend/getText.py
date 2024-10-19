import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from pydantic import BaseModel
from typing import List
import importlib.util

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ROI(BaseModel):
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    # Save the uploaded PDF to a temporary file
    temp_pdf_file = "temp_pdf_file.pdf"
    with open(temp_pdf_file, "wb") as buffer:
        buffer.write(await file.read())
    
    # Load the ROI coordinates from the Python file
    roi_coordinates_path = "backend/output2/roi_coordinates.py"
    spec = importlib.util.spec_from_file_location("roi_coordinates", roi_coordinates_path)
    roi_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roi_module)

    roi_list = roi_module.roi_list  # This contains the list of ROI

    # Open the PDF and extract text
    pdf_document = fitz.open(temp_pdf_file)
    page = pdf_document[0]  # Assuming you want the first page
    extracted_text = {}

    # Iterate through the ROI and extract text
    for roi in roi_list:
        rect = fitz.Rect(roi['x1'], roi['y1'], roi['x2'], roi['y2'])
        text = page.get_text("text", clip=rect)
        extracted_text[roi['name']] = text.strip()  # Save extracted text with ROI name as key

    pdf_document.close()
    os.remove(temp_pdf_file)  # Clean up the temporary file
    return extracted_text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
