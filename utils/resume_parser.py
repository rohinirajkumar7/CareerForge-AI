from pypdf import PdfReader
from docx import Document
import io


def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_text_from_docx(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    document = Document(io.BytesIO(file_bytes))

    text = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)

    return "\n".join(text)


def extract_resume_text(uploaded_file):

    file_type = uploaded_file.name.lower()

    if file_type.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    elif file_type.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)

    else:
        raise ValueError("Unsupported resume format.")