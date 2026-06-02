import os
import re
from pypdf import PdfReader

class DocumentLoader:
    """
    Responsible for loading documents (PDFs) from the data directory
    and performing text cleaning.
    """
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)

    def clean_text(self, text: str) -> str:
        """
        Cleans text by removing:
        - HTML tags
        - Multiple sequential spaces/newlines/tabs
        - Leading/trailing whitespace
        """
        # Remove HTML tags (if any)
        text = re.sub(r'<[^>]*>', '', text)
        
        # Remove multiple whitespaces/tabs/newlines and replace with a single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def load_pdf(self, file_path: str) -> str:
        """
        Extracts text page by page from a PDF and returns the cleaned text.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        reader = PdfReader(file_path)
        full_text = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                cleaned = self.clean_text(text)
                full_text.append(cleaned)
                
        return "\n".join(full_text)

    def load_all_documents(self) -> str:
        """
        Loads all PDF documents from the data directory and combines them.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
        if not files:
            raise FileNotFoundError(f"No PDF files found in {self.data_dir}. Please place documents there.")
            
        combined_text = []
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            print(f"Loading document: {file}")
            doc_text = self.load_pdf(file_path)
            combined_text.append(doc_text)
            
        return "\n".join(combined_text)

if __name__ == "__main__":
    # Test loader local run
    loader = DocumentLoader("./data")
    try:
        text = loader.load_all_documents()
        print(f"Loaded {len(text)} characters of text.")
    except Exception as e:
        print(f"Loader execution error: {e}")
