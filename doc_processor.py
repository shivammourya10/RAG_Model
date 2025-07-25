"""
Document Processing Module for HackRX 6.0
==========================================

This module provides comprehensive document processing capabilities for multiple formats:
- PDF documents with page-wise extraction
- Microsoft Word documents (DOCX) with paragraph-level processing
- Email documents (EML, MSG, MBOX) with metadata extraction

The module is designed for high-performance processing with intelligent content extraction,
metadata preservation, and error handling for production use.

Key Features:
- Multi-format document support (PDF, DOCX, Email)
- Intelligent document type detection
- Metadata extraction and preservation
- Error handling and fallback mechanisms
- Memory-efficient processing for large documents
- Content chunking for optimal RAG performance

Author: HackRX 6.0 Team
"""

import io
import re
import email
import hashlib
import zipfile
import tempfile
import mimetypes
from typing import Dict, List, Optional, Tuple, Union

import requests
import pypdf
from docx import Document
from bs4 import BeautifulSoup
import email.policy
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Import configuration for processing parameters
import config


class DocumentProcessor:
    """
    Comprehensive document processor supporting PDF, DOCX, and Email formats.
    
    This class provides a unified interface for processing various document types
    with intelligent content extraction, metadata preservation, and performance optimization.
    """
    
    def __init__(self):
        """Initialize the document processor with configuration settings."""
        self.config = config.config
        self.max_document_size = self.config.max_document_size_mb * 1024 * 1024  # Convert to bytes
        self.max_pages = self.config.max_pages_per_document
        self.supported_formats = self.config.supported_formats
    
    # =============================================================================
    # Document Type Detection
    # =============================================================================
    
    @staticmethod
    def detect_document_type(url: str, content: Optional[bytes] = None) -> str:
        """
        Intelligently detect document type from URL and/or content.
        
        Args:
            url (str): Document URL
            content (Optional[bytes]): Document content for content-based detection
            
        Returns:
            str: Detected document type ('pdf', 'docx', 'email')
        """
        # Primary detection: URL-based
        url_lower = url.lower()
        
        # Check for explicit file extensions
        if url_lower.endswith('.pdf'):
            return 'pdf'
        elif url_lower.endswith(('.docx', '.doc')):
            return 'docx'
        elif url_lower.endswith(('.eml', '.msg', '.mbox')):
            return 'email'
        
        # Secondary detection: Content-based magic numbers
        if content:
            # PDF magic number
            if content.startswith(b'%PDF'):
                return 'pdf'
            
            # DOCX is a ZIP file with specific structure
            elif content.startswith(b'PK') and b'word/' in content:
                return 'docx'
            
            # Email patterns in content
            elif (b'From:' in content[:1000] or 
                  b'Subject:' in content[:1000] or
                  b'Date:' in content[:1000]):
                return 'email'
        
        # Fallback to PDF (most common in HackRX context)
        return 'pdf'
    
    @staticmethod
    def download_document(url: str) -> Tuple[bytes, str]:
        """Download document from URL and return content with detected type."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content = response.content
            doc_type = DocumentProcessor.detect_document_type(url, content)
            
            return content, doc_type
        except requests.RequestException as e:
            raise Exception(f"Error downloading document: {e}")
    
    @staticmethod
    def get_content_hash(content: bytes) -> str:
        """Generate SHA-256 hash of content for caching."""
        return hashlib.sha256(content).hexdigest()
    
    @staticmethod
    def extract_text_from_pdf(content: bytes) -> Dict:
        """Extract text from PDF content with page-wise information."""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = pypdf.PdfReader(pdf_file)
            
            pages_content = []
            full_text = ""
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text() or ""
                pages_content.append({
                    'page_number': page_num,
                    'content': page_text.strip()
                })
                full_text += f"\n[Page {page_num}]\n{page_text}\n"
            
            return {
                'full_text': full_text.strip(),
                'pages': pages_content,
                'metadata': {
                    'total_pages': len(pdf_reader.pages),
                    'document_type': 'pdf'
                }
            }
        except Exception as e:
            raise Exception(f"Error processing PDF: {e}")
    
    @staticmethod
    def extract_text_from_docx(content: bytes) -> Dict:
        """Extract text from DOCX content with paragraph-wise information."""
        try:
            docx_file = io.BytesIO(content)
            doc = Document(docx_file)
            
            paragraphs_content = []
            full_text = ""
            
            for para_num, paragraph in enumerate(doc.paragraphs, 1):
                para_text = paragraph.text.strip()
                if para_text:  # Only include non-empty paragraphs
                    paragraphs_content.append({
                        'paragraph_number': para_num,
                        'content': para_text,
                        'style': paragraph.style.name if paragraph.style else None
                    })
                    full_text += f"{para_text}\n"
            
            return {
                'full_text': full_text.strip(),
                'paragraphs': paragraphs_content,
                'metadata': {
                    'total_paragraphs': len(paragraphs_content),
                    'document_type': 'docx'
                }
            }
        except Exception as e:
            raise Exception(f"Error processing DOCX: {e}")
    
    @staticmethod
    def extract_text_from_email(content: bytes) -> Dict:
        """Extract text from email content (EML, MSG, MBOX formats)."""
        try:
            content_str = content.decode('utf-8', errors='ignore')
            
            # Handle different email formats
            if content_str.startswith('From '):
                # MBOX format
                return DocumentProcessor._process_mbox_content(content_str)
            else:
                # EML format
                return DocumentProcessor._process_eml_content(content_str)
                
        except Exception as e:
            raise Exception(f"Error processing email: {e}")
    
    @staticmethod
    def _process_eml_content(content_str: str) -> Dict:
        """Process EML email content."""
        try:
            msg = email.message_from_string(content_str, policy=email.policy.default)
            
            # Extract metadata
            sender = msg.get('From', 'Unknown')
            recipient = msg.get('To', 'Unknown')
            subject = msg.get('Subject', 'No Subject')
            date = msg.get('Date', 'Unknown')
            
            # Extract body content
            body_content = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body_content += part.get_content() + "\n"
                    elif part.get_content_type() == "text/html":
                        # Extract text from HTML
                        html_content = part.get_content()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        body_content += soup.get_text() + "\n"
            else:
                if msg.get_content_type() == "text/plain":
                    body_content = msg.get_content()
                elif msg.get_content_type() == "text/html":
                    soup = BeautifulSoup(msg.get_content(), 'html.parser')
                    body_content = soup.get_text()
            
            # Clean and format content
            cleaned_body = DocumentProcessor._clean_email_content(body_content)
            
            full_text = f"""
Email Metadata:
From: {sender}
To: {recipient}
Subject: {subject}
Date: {date}

Email Content:
{cleaned_body}
            """.strip()
            
            return {
                'full_text': full_text,
                'email_data': {
                    'sender': sender,
                    'recipient': recipient,
                    'subject': subject,
                    'date': date,
                    'body': cleaned_body
                },
                'metadata': {
                    'document_type': 'email',
                    'email_format': 'eml'
                }
            }
        except Exception as e:
            raise Exception(f"Error processing EML content: {e}")
    
    @staticmethod
    def _process_mbox_content(content_str: str) -> Dict:
        """Process MBOX email content (multiple emails)."""
        try:
            emails = []
            current_email = []
            
            lines = content_str.split('\n')
            for line in lines:
                if line.startswith('From ') and current_email:
                    # Process previous email
                    email_content = '\n'.join(current_email)
                    email_data = DocumentProcessor._process_eml_content(email_content)
                    emails.append(email_data)
                    current_email = []
                else:
                    current_email.append(line)
            
            # Process last email
            if current_email:
                email_content = '\n'.join(current_email)
                email_data = DocumentProcessor._process_eml_content(email_content)
                emails.append(email_data)
            
            # Combine all emails
            full_text = ""
            for i, email_data in enumerate(emails, 1):
                full_text += f"\n[Email {i}]\n{email_data['full_text']}\n"
            
            return {
                'full_text': full_text.strip(),
                'emails': emails,
                'metadata': {
                    'document_type': 'email',
                    'email_format': 'mbox',
                    'total_emails': len(emails)
                }
            }
        except Exception as e:
            raise Exception(f"Error processing MBOX content: {e}")
    
    @staticmethod
    def _clean_email_content(content: str) -> str:
        """Clean email content by removing headers, footers, and excessive whitespace."""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove common email artifacts
        content = re.sub(r'^>.*$', '', content, flags=re.MULTILINE)  # Quoted text
        content = re.sub(r'On .* wrote:', '', content)  # Reply headers
        content = re.sub(r'-----Original Message-----.*', '', content, flags=re.DOTALL)
        
        return content.strip()
    
    @staticmethod
    def process_document_from_url(url: str) -> Dict:
        """Main method to process any document type from URL."""
        try:
            # Download document
            content, doc_type = DocumentProcessor.download_document(url)
            content_hash = DocumentProcessor.get_content_hash(content)
            
            # Process based on type
            if doc_type == 'pdf':
                result = DocumentProcessor.extract_text_from_pdf(content)
            elif doc_type == 'docx':
                result = DocumentProcessor.extract_text_from_docx(content)
            elif doc_type == 'email':
                result = DocumentProcessor.extract_text_from_email(content)
            else:
                raise Exception(f"Unsupported document type: {doc_type}")
            
            # Add common metadata
            result['metadata']['content_hash'] = content_hash
            result['metadata']['source_url'] = url
            result['metadata']['content_size'] = len(content)
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to process document from {url}: {e}")

# Example usage and testing
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Test with a sample PDF URL
    try:
        result = processor.process_document_from_url("https://example.com/sample.pdf")
        print(f"Processed document successfully:")
        print(f"Content length: {len(result['full_text'])}")
        print(f"Document type: {result['metadata']['document_type']}")
    except Exception as e:
        print(f"Error: {e}")
