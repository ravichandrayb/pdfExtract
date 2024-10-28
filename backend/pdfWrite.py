import fitz
import sys
from typing import Dict, List

# Your coordinates
roi_list = [{'name': 'pos', 'type': 'text', 'x1': 241, 'y1': 65, 'x2': 395, 'y2': 88}]

def insert_image_to_pdf(rect,pdf_path, image_path, page_number=0,):
    """
    Insert an image into a PDF file at specified location.
    
    Args:
        pdf_path (str): Path to the PDF file
        image_path (str): Path to the image file
        page_number (int): Page number where to insert the image (0-based index)
        location (tuple): (x, y) coordinates for image placement
        zoom (float): Zoom factor for the image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        # Check if page number is valid
        if page_number >= len(pdf_document):
            raise ValueError(f"Page number {page_number} is out of range")
        
        # Get the specified page
        page = pdf_document[page_number]
        
        # Insert image
        
                                  
        
        # Insert the image
        page.insert_image(rect, filename=image_path)
        
        # Save the modified PDF
        output_path = f"modified_{os.path.basename(pdf_path)}"
        pdf_document.save(output_path)
        pdf_document.close()
        
        return True
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False
    
def write_to_pdf(input_pdf: str, output_pdf: str, values: Dict[str, str], roi_list: List[Dict]) -> None:
    """
    Write values to specific coordinates on a PDF file using bounding box coordinates.
    
    Args:
        input_pdf (str): Path to input PDF file
        output_pdf (str): Path to save the modified PDF
        values (dict): Dictionary of field names and values to write
        roi_list (list): List of dictionaries containing coordinate information
    """
    try:
        # Open the PDF
        doc = fitz.open(input_pdf)
        page = doc[0]  # Assuming we're working with the first page
        
        # Create a text writer
        text_writer = fitz.TextWriter(page.rect)
        
        # Add each value at its corresponding coordinates
        for roi in roi_list:
            field_name = roi['name']
            if field_name in values:
                # Calculate the center position for text placement
                x = (roi['x1'] + roi['x2']) / 2
                y = (roi['y1'] + roi['y2']) / 2
                rect = fitz.Rect(roi['x1'], roi['y1'], roi['x2'], roi['y2'])
                
                # Clear existing content in the rectangle
                # This adds a white rectangle to cover existing content
                page.draw_rect(rect, color=fitz.utils.getColor("white"), fill=fitz.utils.getColor("white"))
                # Calculate font size based on box height
                # Assuming we want the text height to be about 70% of the box height
                font_size = 6.96
                
                # Create font
                with open("ArialMT.ttf", "rb") as font_file:
                    font_buffer = font_file.read()

# Create font from buffer
                font = fitz.Font(fontbuffer=font_buffer)
                
                
                # Get the text value
                text = values[field_name]
                if(text == True):
                    insert_image_to_pdf(rect,input_pdf, "/checked_checkbox.png", page_number=0, )
                if(text == False):
                    insert_image_to_pdf(rect,input_pdf, "/unchecked_checkbox.png", page_number=0 )
                # Add text to the writer
                text_writer.append(
                    (roi['x1'], roi['y1']+15),
                    text,
                    font=font,
                    fontsize=font_size,
                )
        
        # Apply the text to the page
        text_writer.write_text(page)
        
        # Save the modified PDF
        doc.save(output_pdf)
        doc.close()
        print(f"Successfully wrote to {output_pdf}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

def get_user_input(roi_list: List[Dict]) -> Dict[str, str]:
    """
    Get user input for each field defined in roi_list.
    
    Args:
        roi_list (list): List of dictionaries containing field information
    
    Returns:
        dict: Dictionary of field names and their values
    """
    values = {}
    print("Please enter values for each field:")
    for roi in roi_list:
        if roi['type'] == 'text':  # Only process text fields
            field_name = roi['name']
            value = input(f"Enter value for {field_name}: ")
            values[field_name] = value
    return values

def main():
    # Define input and output PDF paths
    input_pdf = "./uploads/input.pdf"
    output_pdf = "output.pdf"
    
    # Get values from user
    values = get_user_input(roi_list)
    
    # Write values to PDF
    write_to_pdf(input_pdf, output_pdf, values, roi_list)

if __name__ == "__main__":
    main()