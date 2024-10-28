import fitz

def get_text_properties(pdf_path, x1, y1, x2, y2):
    """
    Extract text properties including font information from a specific area of the PDF.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]  # Assuming first page
    
    # Create rectangle for the area of interest
    rect = fitz.Rect(x1, y1, x2, y2)
    
    # Extract text dictionary with properties
    text_dict = page.get_text("dict", clip=rect)
    
    # Print detailed information about each text block
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                print("\nText:", span.get("text"))
                print("Font:", span.get("font"))
                print("Font size:", span.get("size"))
                print("Color:", span.get("color"))
                print("Flags:", span.get("flags"))
    
    doc.close()

# Using your ROI coordinates
x1, y1, x2, y2 = 241, 65, 395, 88  # Your coordinates
pdf_path = "./uploads/input.pdf"


try:
    get_text_properties(pdf_path, x1, y1, x2, y2)
except Exception as e:
    print(f"An error occurred: {str(e)}")