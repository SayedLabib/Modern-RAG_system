"""
Advanced text preprocessing for Bangla content
"""
import re
import os
import unicodedata
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class BanglaTextPreprocessor:
    """Advanced Bangla text preprocessing for better RAG performance"""
    
    def __init__(self):
        # Common OCR/encoding error patterns in Bangla text
        self.error_patterns = {
            'র্ি': 'রি',
            'ব্জ': 'বল',
            'কক': 'কে', 
            'র্খ': 'খ',
            'র্দ': 'দে',
            'র্ব': 'ব',
            'র্ন': 'ন',
            'র্ত': 'ত',
            'যকান': 'কোন',
            'যদর্খ': 'দেখ',
            'হইল': 'হলো',
            'কর্ি': 'করি',
            'তাহা': 'তা',
            'সককল': 'সকল',
            'যসই': 'সেই',
            'উহা': 'এটা',
            'পকি': 'পর',
            'হাকত': 'হাতে',
            'মাকক': 'মাকে',
            'লই া': 'নিয়ে',
            'দদক': 'দেখ',
            'যব্': 'বে',
            'িম্ভু': 'শম্ভু',
        }
        
        # Bangla punctuation normalization
        self.punctuation_map = {
            '।': '।',  # Bangla danda
            '?': '?',
            '!': '!',
            ',': ',',
            ';': ';',
            ':': ':',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Bangla text"""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Fix common OCR/encoding errors
        for error, correction in self.error_patterns.items():
            text = text.replace(error, correction)
        
        # Normalize punctuation
        for old_punct, new_punct in self.punctuation_map.items():
            text = text.replace(old_punct, new_punct)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix sentence boundaries
        text = re.sub(r'।\s*([া-৯])', r'। \1', text)  # Add space after danda
        text = re.sub(r'([া-৯])\s*।', r'\1।', text)   # Remove space before danda
        
        # Remove numbers mixed with text inappropriately
        text = re.sub(r'(\d+)([া-৯])', r'\1 \2', text)
        text = re.sub(r'([া-৯])(\d+)', r'\1 \2', text)
        
        # Clean up multiple punctuation
        text = re.sub(r'[।]{2,}', '।', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[!]{2,}', '!', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into proper sentences"""
        # Split on Bangla sentence endings
        sentences = re.split(r'[।?!]+', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def create_semantic_chunks(self, text: str, chunk_size: int = 600, 
                             overlap: int = 50) -> List[dict]:
        """Create semantically meaningful chunks"""
        sentences = self.split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'content': current_chunk.strip(),
                    'length': current_length,
                    'sentence_count': current_chunk.count('।') + current_chunk.count('?') + current_chunk.count('!')
                })
                
                # Start new chunk with overlap
                if overlap > 0 and len(chunks) > 0:
                    # Take last few words for overlap
                    words = current_chunk.split()
                    overlap_words = words[-overlap:] if len(words) > overlap else words
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'length': current_length,
                'sentence_count': current_chunk.count('।') + current_chunk.count('?') + current_chunk.count('!')
            })
        
        return chunks
    
    def separate_content_types(self, text: str) -> Tuple[str, str, str]:
        """Separate different types of content"""
        narrative_text = ""
        questions = ""
        metadata = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect questions (lines ending with ?)
            if line.endswith('?') or 'কি' in line or 'কেন' in line or 'কোথায়' in line:
                questions += line + '\n'
            # Detect metadata/reference lines
            elif re.match(r'^\d+\.|^[ক-ঘ]\.|^[a-d]\.', line):
                metadata += line + '\n'
            # Everything else is narrative
            else:
                narrative_text += line + ' '
        
        return narrative_text.strip(), questions.strip(), metadata.strip()
    
    def process_file(self, file_path: str, output_dir: str = None) -> dict:
        """Process a complete file and create clean versions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            logger.info(f"Processing file: {file_path}")
            logger.info(f"Original text length: {len(raw_text)} characters")
            
            # Clean the text
            cleaned_text = self.clean_text(raw_text)
            logger.info(f"Cleaned text length: {len(cleaned_text)} characters")
            
            # Separate content types
            narrative, questions, metadata = self.separate_content_types(cleaned_text)
            
            # Create semantic chunks for narrative text
            narrative_chunks = self.create_semantic_chunks(narrative)
            
            # Create chunks for questions (smaller chunks)
            question_chunks = self.create_semantic_chunks(questions, chunk_size=300, overlap=20)
            
            results = {
                'original_length': len(raw_text),
                'cleaned_length': len(cleaned_text),
                'narrative_chunks': narrative_chunks,
                'question_chunks': question_chunks,
                'metadata': metadata,
                'total_chunks': len(narrative_chunks) + len(question_chunks)
            }
            
            # Save processed versions if output directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Save cleaned full text
                with open(os.path.join(output_dir, f"{base_name}_cleaned.txt"), 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                # Save narrative chunks
                with open(os.path.join(output_dir, f"{base_name}_narrative_chunks.txt"), 'w', encoding='utf-8') as f:
                    for i, chunk in enumerate(narrative_chunks):
                        f.write(f"--- Chunk {i+1} ---\n")
                        f.write(chunk['content'])
                        f.write(f"\n(Length: {chunk['length']}, Sentences: {chunk['sentence_count']})\n\n")
                
                # Save questions
                if questions:
                    with open(os.path.join(output_dir, f"{base_name}_questions.txt"), 'w', encoding='utf-8') as f:
                        f.write(questions)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return {}

def main():
    """Test the preprocessor"""
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = BanglaTextPreprocessor()
    
    # Process the current data file
    input_file = "app/data/Bangla_cleaned_book.txt"
    output_dir = "app/data/processed"
    
    if os.path.exists(input_file):
        results = preprocessor.process_file(input_file, output_dir)
        
        print("\n=== Processing Results ===")
        print(f"Original length: {results.get('original_length', 0):,} characters")
        print(f"Cleaned length: {results.get('cleaned_length', 0):,} characters")
        print(f"Total chunks created: {results.get('total_chunks', 0)}")
        print(f"Narrative chunks: {len(results.get('narrative_chunks', []))}")
        print(f"Question chunks: {len(results.get('question_chunks', []))}")
        
        # Show sample of improved chunks
        print("\n=== Sample Improved Chunks ===")
        narrative_chunks = results.get('narrative_chunks', [])
        for i, chunk in enumerate(narrative_chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Content: {chunk['content'][:200]}...")
            print(f"Length: {chunk['length']}, Sentences: {chunk['sentence_count']}")
    else:
        print(f"File not found: {input_file}")

if __name__ == "__main__":
    main()
