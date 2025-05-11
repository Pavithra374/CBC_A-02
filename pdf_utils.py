import fitz  # PyMuPDF
import base64
from io import BytesIO
import tempfile
import os
from gtts import gTTS
import time

def get_pdf_display(pdf_page):
    """Convert PDF page to displayable format"""
    pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("png")
    img_b64 = base64.b64encode(img_bytes).decode()
    return f"data:image/png;base64,{img_b64}"

def extract_sentences(page):
    """Extract sentences from PDF page"""
    text = page.get_text()
    # Simple sentence splitting - can be improved
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sentences

def text_to_speech(text, output_dir):
    """Convert text to speech and save as audio file"""
    # Create a temporary filename
    timestamp = int(time.time())
    filename = f"{output_dir}/speech_{timestamp}.mp3"
    
    # Generate speech
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    
    return filename

def process_pdf(pdf_file, page_num=0):
    """Process a PDF file to extract text and create page image"""
    try:
        # Create a temporary file to save the PDF
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf.write(pdf_file.read())
        temp_pdf.close()
        
        # Open PDF with PyMuPDF
        doc = fitz.open(temp_pdf.name)
        
        if page_num >= len(doc):
            page_num = 0
            
        # Get the specified page
        page = doc[page_num]
        
        # Get image for display
        img_data = get_pdf_display(page)
        
        # Get text content
        sentences = extract_sentences(page)
        
        # Store total pages before closing
        total_pages = len(doc)
        
        # Close document and remove temp file
        doc.close()
        os.unlink(temp_pdf.name)
        
        return {
            'success': True,
            'image': img_data,
            'sentences': sentences,
            'total_pages': total_pages
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
