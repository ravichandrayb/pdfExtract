import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi.staticfiles import StaticFiles
import importlib.util
from PIL import Image
import numpy as np
import cv2
from fastapi.responses import FileResponse
from fastapi.background import BackgroundTasks
from werkzeug.utils import secure_filename
import logging
from pathlib import Path
import shutil
import uuid
import tempfile
from datetime import datetime
import asyncio
from enum import Enum
import traceback
import re

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()
current_dir = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def cleanup_files(file_paths: List[str]):
    """Delete temporary files after response has been sent"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {e}")

# Define UPLOAD_FOLDER
UPLOAD_FOLDER = os.path.join(current_dir, "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Mount the static directory
app.mount("/static", StaticFiles(directory=current_dir), name="static")

def is_checkbox_checked(page, rect, method='pixel_density', debug=False):
    """
    Analyze the checkbox area to determine if it's checked, after removing the border.
    
    Args:
        page (fitz.Page): PDF page object
        rect (fitz.Rect): Rectangle defining the checkbox area
        method (str): Detection method to use ('pixel_density', 'contour', or 'combined')
        debug (bool): If True, returns additional detection details
        
    Returns:
        bool: True if checkbox is checked, False otherwise
        dict: Debug information (if debug=True)
    """
    # Get the pixel data for the checkbox area
    zoom_x = 5.0  # horizontal zoom
    zoom_y = 5.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
    pix = page.get_pixmap(matrix=mat, clip=rect)  # use 'mat' instead of the identity matrix
    # print(pix)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_gray = img.convert('L')
    pixels = np.array(img_gray)
    
    # Remove the border
    def remove_border(img_array):
        # Convert to binary
        _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img_array
            
        # Find the largest contour (likely the checkbox border)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask for the border
        border_mask = np.zeros_like(img_array)
        cv2.drawContours(border_mask, [largest_contour], -1, (255, 255, 255), 1)
        
        # Create a slightly smaller mask to remove internal border pixels
        kernel = np.ones((3,3), np.uint8)
        border_mask = cv2.dilate(border_mask, kernel, iterations=1)
        
        # Remove border pixels from original image
        result = img_array.copy()
        result[border_mask > 0] = 255  # Set border pixels to white
        
        # Create content mask (exclude borders and padding)
        h, w = img_array.shape
        padding = max(4, int(min(h, w) * 0.2))  # Dynamic padding based on size
        content_mask = np.zeros_like(img_array)
        content_mask[padding:-padding, padding:-padding] = 1
        
        # Apply content mask
        result = result * content_mask + 255 * (1 - content_mask)
        
        return result
    
    # Remove border from image
    pixels_no_border = remove_border(pixels)
    print(pixels_no_border)
    debug_info = {}
    
    if method == 'pixel_density' or method == 'combined':
        # Method 1: Pixel density analysis
        dark_threshold = 222
        # Calculate percentage only in the inner region
        h, w = pixels_no_border.shape
        padding = max(2, int(min(h, w) * 0.2))
        inner_region = pixels_no_border[padding:-padding, padding:-padding]
        print(inner_region.size)
        print("sum",np.sum(inner_region < dark_threshold))
        print("inner region",inner_region.size)
        # Check if inner_region is not empty
        if inner_region.size > 0:
            dark_pixel_percentage = np.sum(inner_region < dark_threshold) / inner_region.size
        else:
            dark_pixel_percentage = 0  # Default to 0 if inner region is empty
        
        is_checked_density = dark_pixel_percentage > 0.2  # Lower threshold since we removed border
        debug_info['dark_pixel_percentage'] = float(dark_pixel_percentage)
        
        if method == 'pixel_density':
            if debug:
                debug_info['processed_image'] = pixels_no_border
                return bool(is_checked_density), debug_info
            return bool(is_checked_density)
    
    if method == 'contour' or method == 'combined':
        # Method 2: Contour detection on border-removed image
        _, binary = cv2.threshold(pixels_no_border, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours
        significant_contours = []
        min_contour_area = pixels_no_border.size * 0.015  # Lower threshold since we removed border
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                significant_contours.append(contour)
        
        is_checked_contour = len(significant_contours) > 0
        debug_info['significant_contours'] = len(significant_contours)
        
        if method == 'contour':
            if debug:
                debug_info['processed_image'] = pixels_no_border
                return bool(is_checked_contour), debug_info
            return bool(is_checked_contour)
    
    # Combined method: Use both techniques
    if method == 'combined':
        is_checked = is_checked_density or is_checked_contour
        if debug:
            debug_info['processed_image'] = pixels_no_border
            return bool(is_checked), debug_info
        return bool(is_checked)
    
    raise ValueError(f"Unknown method: {method}")

# def is_checkbox_checked_old(page, rect, method='combined', debug=False):
    
#     # pix = page.get_pixmap(clip=rect)
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     img_gray = img.convert('L')
#     pixels = np.array(img_gray)
#     print(pixels,pixels.size,np.sum(pixels < 255))
    
#     debug_info = {}
    
#     if method == 'pixel_density' or method == 'combined':
#         # Method 1: Pixel density analysis
#         dark_threshold = 255
#         dark_pixel_percentage = np.sum(pixels < dark_threshold) / pixels.size
#         print(dark_pixel_percentage)
#         is_checked_density = bool(dark_pixel_percentage > 0.1)  # Convert to Python bool
#         debug_info['dark_pixel_percentage'] = float(dark_pixel_percentage)  # Convert to float
        
#         if method == 'pixel_density':
#             if debug:
#                 return is_checked_density, debug_info
#             return is_checked_density
    
#     if method == 'contour' or method == 'combined':
#         # Method 2: Contour detection
#         # Convert to binary image
#         _, binary = cv2.threshold(pixels, 127, 255, cv2.THRESH_BINARY_INV)
        
#         # Find contours
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Analyze contours
#         significant_contours = []
#         min_contour_area = pixels.size * 0.02  # Minimum 2% of checkbox area
        
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area > min_contour_area:
#                 significant_contours.append(contour)
        
#         is_checked_contour = bool(len(significant_contours) > 0)  # Convert to Python bool
#         debug_info['significant_contours'] = len(significant_contours)
        
#         if method == 'contour':
#             if debug:
#                 return is_checked_contour, debug_info
#             return is_checked_contour
    
#     # Combined method: Use both techniques
#     if method == 'combined':
#         print(f"Density check: {is_checked_density}, Contour check: {is_checked_contour}")
#         is_checked = bool(is_checked_density or is_checked_contour)  # Ensure Python bool
#         if debug:
#             return is_checked, debug_info
#         return is_checked
    
#     raise ValueError(f"Unknown method: {method}")
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ROI(BaseModel):
    name: str
    x1: int
    y1: int
    x2: int
    y2: int




logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



# Add the new PDFSplitter class
class PDFSplitter:
    @staticmethod
    async def split_pdf(file_path: str, output_dir: str) -> List[str]:
        """
        Split a PDF file into individual pages.
        
        Args:
            file_path (str): Path to the input PDF file
            output_dir (str): Directory to save the split PDF files
            
        Returns:
            List[str]: List of paths to the created PDF files
        """
        created_files = []
        
        try:
            # Open the PDF file
            pdf_document = fitz.open(file_path)
            
            if not pdf_document.is_pdf:
                raise ValueError("Uploaded file is not a valid PDF")
            
            # Iterate through each page
            for page_num in range(len(pdf_document)):
                # Create a new PDF with just this page
                new_pdf = fitz.open()
                new_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
                
                # Generate output filename
                output_path = os.path.join(output_dir, f"page_{page_num + 1}.pdf")
                
                # Save the new PDF
                new_pdf.save(output_path)
                created_files.append(output_path)
                
                # Close the new PDF
                new_pdf.close()
                
            return created_files
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
            
        finally:
            if 'pdf_document' in locals():
                pdf_document.close()

# Add new imports and configurations for PDF compression
class CompressionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class PDFCompressor:
    @staticmethod
    def get_compression_params(level: CompressionLevel) -> dict:
        """
        Get compression parameters based on the selected level.
        """
        params = {
            CompressionLevel.LOW: {
                "deflate": True,
                "garbage": 1,
                "clean": True
            },
            CompressionLevel.MEDIUM: {
                "deflate": True,
                "garbage": 2,
                "clean": True,
                "linear": True
            },
            CompressionLevel.HIGH: {
                "deflate": True,
                "garbage": 4,
                "clean": True,
                "linear": True,
                "ascii": True
            }
        }
        return params[level]

    @staticmethod
    async def compress_pdf(
        input_path: str,
        output_path: str,
        compression_level: CompressionLevel,
        image_quality: Optional[int] = 80
    ) -> tuple[float, float]:
        """
        Compress a PDF file with specified compression level and image quality.
        """
        try:
            # Get original file size
            original_size = os.path.getsize(input_path) / (1024 * 1024)  # Convert to MB

            # Open the PDF
            pdf_document = fitz.open(input_path)
            
            if not pdf_document.is_pdf:
                raise ValueError("Invalid PDF file")

            # Process each page to compress images
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Get all images on the page
                image_list = page.get_images()
                
                for img_index, image in enumerate(image_list):
                    xref = image[0]
                    
                    # Check if image exists and hasn't been processed
                    if xref > 0 and not pdf_document.xref_is_compressed(xref):
                        # Try to compress the image
                        try:
                            pdf_document.extract_image(xref)
                            # Note: In production, you might want to actually process
                            # the image using PIL or similar to reduce quality
                        except Exception:
                            continue

            # Get compression parameters
            compression_params = PDFCompressor.get_compression_params(compression_level)
            
            # Save with compression
            pdf_document.save(
                output_path,
                **compression_params
            )
            
            # Get compressed file size
            compressed_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
            
            return original_size, compressed_size

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")
            
        finally:
            if 'pdf_document' in locals():
                pdf_document.close()

@app.post("/merge-pdfs")
async def merge_pdfs(
    pdfs: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    temp_files = []
    try:
        logger.debug(f"Received {len(pdfs)} PDFs for merging")
        
        # Validate number of files
        if len(pdfs) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 PDF files")
        
        # Create a new PDF document
        merged_pdf = fitz.open()
        
        # Process each uploaded file
        for pdf in pdfs:
            logger.debug(f"Processing file: {pdf.filename}")
            
            # Create a unique filename for each uploaded file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{pdf.filename}")
            temp_files.append(temp_path)
            
            # Save uploaded file
            logger.debug(f"Saving file to: {temp_path}")
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(pdf.file, buffer)
            
            # Open and merge the PDF
            try:
                logger.debug(f"Opening and merging PDF: {temp_path}")
                with fitz.open(temp_path) as source:
                    merged_pdf.insert_pdf(source)
            except Exception as e:
                logger.error(f"Error processing {pdf.filename}: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing {pdf.filename}: {str(e)}"
                )
        
        # Generate unique output filename
        output_filename = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Save the merged PDF
        logger.debug(f"Saving merged PDF to: {output_path}")
        merged_pdf.save(output_path)
        merged_pdf.close()
        
        # Add output file to cleanup list
        temp_files.append(output_path)
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(cleanup_files, temp_files)
        
        logger.debug("Returning merged PDF")
        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename="merged.pdf",
            background=background_tasks
        )
            
    except Exception as e:
        logger.error(f"Error in merge_pdfs: {str(e)}")
        logger.error(traceback.format_exc())
        # Clean up any temporary files if an error occurs
        cleanup_files(temp_files)
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Cleanup endpoint for maintenance
@app.post("/cleanup")
async def cleanup_temporary_files():
    """Cleanup any leftover temporary files"""
    try:
        # Clean upload folder
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        # Clean output folder
        for file in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        return {"message": "Cleanup completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(await file.read())
    
    pdf_document = fitz.open(temp_file)
    page = pdf_document[0]
    pix = page.get_pixmap()
    img_path = "temp_page.png"
    pix.save(img_path)
    
    pdf_document.close()
    os.remove(temp_file)
    
    return {"image_path": img_path}

# Enable CORS
class ROI(BaseModel):
    name: str
    type: str
    x1: int
    y1: int
    x2: int
    y2: int

@app.post("/save-roi")
async def save_roi(roi_list: List[ROI]):
    print(roi_list)
    # Create output2 folder if it doesn't exist
    if not os.path.exists('output2'):
        os.makedirs('output2')
    
    # Save ROI list to a file
    with open('output2/roi_coordinates.py', 'w') as f:
        
        f.write("roi_list = " + str([roi.dict() for roi in roi_list]))
    
    return {"message": "ROI coordinates saved successfully"}

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    # Save the uploaded PDF to a temporary file
    
    # Save the uploaded PDF to a temporary file
    temp_pdf_file = "temp_pdf_file.pdf"
    with open(temp_pdf_file, "wb") as buffer:
        buffer.write(await file.read())
    
    # Load the ROI coordinates
    roi_coordinates_path = "output2/roi_coordinates.py"
    spec = importlib.util.spec_from_file_location("roi_coordinates", roi_coordinates_path)
    roi_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roi_module)

    roi_list = roi_module.roi_list

    # Open the PDF and extract content
    pdf_document = fitz.open(temp_pdf_file)
    page = pdf_document[0]  # Assuming first page
    

    extracted_text = {}

    def get_ordered_text_from_clip(page, rect):
        blocks = page.get_text("dict", clip=rect)["blocks"]
        
        # Extract lines with their positions
        lines = []
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        # Combine all spans in the line
                        text = " ".join(span["text"].replace("\n", " ") for span in line["spans"])
                        if text.strip():
                            lines.append({
                                "text": text.strip(),  # Strip whitespace when creating the dict
                                "y": line["bbox"][1],  # Top y-coordinate
                                "x": line["bbox"][0]   # Left x-coordinate
                            })
        
        # Sort lines primarily by y-coordinate (top to bottom)
        # For lines with very close y-coordinates, sort by x-coordinate
        y_threshold = 5
        
        # Group lines that are close together vertically
        lines.sort(key=lambda x: x["y"])
        grouped_lines = []
        current_group = []
        
        for i, line in enumerate(lines):
            if i == 0:
                current_group.append(line)
                continue
            
            prev_line = lines[i-1]
            if abs(line["y"] - prev_line["y"]) <= y_threshold:
                current_group.append(line)  # Append the dictionary object directly
            else:
                # Sort the current group by x-coordinate
                current_group.sort(key=lambda x: x["x"])
                grouped_lines.extend(current_group)
                current_group = [line]
        
        # Add the last group
        if current_group:
            current_group.sort(key=lambda x: x["x"])
            grouped_lines.extend(current_group)
        
        # Extract just the text in correct order
        ordered_text = " ".join(line["text"] for line in grouped_lines)
        
        return ordered_text.replace("\n", "").replace("  ", " ").strip()

    

    # Iterate through the ROI and extract content
    for roi in roi_list:
        rect = fitz.Rect(roi['x1'], roi['y1'], roi['x2'], roi['y2'])
        
        if roi.get('type') == 'checkbox':
            # Handle checkbox extraction
            is_checked = is_checkbox_checked(page, rect)
            is_checked = bool(is_checked)  # Ensure it's a Python bool
            print(f"Checkbox '{roi['name']}' is_checked: {is_checked}, type: {type(is_checked)}")
            extracted_text[roi['name']] = is_checked
        else:
            # Handle text extraction
            text = get_ordered_text_from_clip(page, rect)
            
            # Split the text into lines
            lines = text.split('\n')
            
            # Remove empty lines and strip whitespace
            lines = [line.strip() for line in lines if line.strip()]
            
            # For addresses, we often want to keep the original order
            # or reverse it (assuming the last line is the most specific)
            if roi['name'] == 'address':
                text_blocks_sorted = list(reversed(lines))
            else:
                text_blocks_sorted = lines
            
            cleaned_text = ''.join(text_blocks_sorted).strip()
            extracted_text[roi['name']] = cleaned_text

    pdf_document.close()
    os.remove(temp_pdf_file)  # Clean up
    print(f"Final extracted_text: {extracted_text}")
    return extracted_text
    
# Add the new endpoint for splitting PDFs


@app.post("/split-pdf/")
async def split_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Create unique directory for this upload
    session_id = str(uuid.uuid4())
    session_dir = Path(UPLOAD_FOLDER) / session_id
    output_dir = session_dir / "split"
    
    try:
        # Create directories
        session_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        temp_path = session_dir / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Split the PDF
        splitter = PDFSplitter()
        split_files = await splitter.split_pdf(str(temp_path), str(output_dir))
        
        # Create a zip file containing all split PDFs
        zip_path = session_dir / "split_pdfs.zip"
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', str(output_dir))
        
        # Ensure the zip file exists
        if not zip_path.exists():
            raise HTTPException(status_code=500, detail="Failed to create zip file")
        
        # Return the zip file
        return FileResponse(
            path=str(zip_path),
            filename=f"split_pdfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            media_type="application/zip"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Schedule cleanup after a delay to ensure file is sent
        asyncio.create_task(delayed_cleanup(session_dir))

async def delayed_cleanup(directory: Path, delay: int = 60):
    """
    Perform delayed cleanup of the temporary directory.
    """
    await asyncio.sleep(delay)
    try:
        shutil.rmtree(str(directory))
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

@app.post("/compress-pdf/")
async def compress_pdf_endpoint(file: UploadFile = File(...), compression_level: int = 2):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Create unique directory for this upload
    session_id = str(uuid.uuid4())
    session_dir = Path(UPLOAD_FOLDER) / session_id
    
    try:
        # Create directory
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        input_path = session_dir / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Compress the PDF
        output_path = session_dir / f"compressed_{file.filename}"
        pdf_document = fitz.open(str(input_path))
        pdf_document.save(str(output_path), garbage=compression_level, deflate=True, clean=True)
        pdf_document.close()
        
        # Return the compressed file
        return FileResponse(
            path=str(output_path),
            filename=f"compressed_{file.filename}",
            media_type="application/pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")
        
    finally:
        # Schedule cleanup after a delay to ensure file is sent
        asyncio.create_task(delayed_cleanup(session_dir))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
