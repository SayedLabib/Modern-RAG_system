"""
LLM generation module using Groq API for response generation
"""
import logging
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass
try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """Result from LLM generation"""
    response: str
    generation_time: float
    token_count: Optional[int] = None
    model_used: str = ""

class GroqGenerator:
    """
    LLM generator using Groq API for text generation
    """
    
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
                 temperature: float = 0.7, max_tokens: int = 2048):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key
            model: Model name to use
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        
        if Groq is None:
            logger.error("Groq package not installed. Install with: pip install groq")
            raise ImportError("Groq package not available")
        
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            logger.warning("No Groq API key provided. Generator will not work.")
    
    def generate_response(self, query: str, context_documents: List[str],
                         system_prompt: Optional[str] = None) -> GenerationResult:
        """
        Generate response using retrieved context
        
        Args:
            query: User query
            context_documents: Retrieved document contents
            system_prompt: Optional system prompt
            
        Returns:
            GenerationResult object
        """
        try:
            if not self.client:
                return GenerationResult(
                    response="Error: Groq API not configured",
                    generation_time=0.0,
                    model_used=self.model
                )
            
            start_time = time.time()
            
            # Build prompt with context
            prompt = self._build_prompt(query, context_documents, system_prompt)
            
            # Generate response
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt or self._get_default_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            generation_time = time.time() - start_time
            
            generated_text = response.choices[0].message.content
            
            # Fix the usage token count access
            token_count = None
            if hasattr(response, 'usage') and response.usage:
                token_count = getattr(response.usage, 'total_tokens', None)
            
            logger.info(f"Generated response in {generation_time:.3f}s using {self.model}")
            
            return GenerationResult(
                response=generated_text,
                generation_time=generation_time,
                token_count=token_count,
                model_used=self.model
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return GenerationResult(
                response=f"Error generating response: {str(e)}",
                generation_time=0.0,
                model_used=self.model
            )
    
    def _build_prompt(self, query: str, context_documents: List[str],
                     system_prompt: Optional[str] = None) -> str:
        """
        Build the prompt with query and context
        
        Args:
            query: User query
            context_documents: Retrieved documents
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        if not context_documents:
            return f"প্রশ্ন: {query}\n\nদয়া করে এই প্রশ্নের উত্তর দিন।"
        
        # Combine context documents
        context = "\n\n".join([f"প্রসঙ্গ {i+1}: {doc}" for i, doc in enumerate(context_documents)])
        
        prompt = f"""নিম্নলিখিত প্রসঙ্গ ব্যবহার করে প্রশ্নের উত্তর দিন:

{context}

প্রশ্ন: {query}

দয়া করে প্রদত্ত প্রসঙ্গের উপর ভিত্তি করে একটি বিস্তারিত এবং সঠিক উত্তর দিন। যদি প্রসঙ্গে যথেষ্ট তথ্য না থাকে, তাহলে তা উল্লেখ করুন।"""

        return prompt
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for the assistant"""
        return """আপনি একজন বিশেষজ্ঞ বাংলা ভাষা বিশ্লেষক এবং সহায়ক সহকারী। আপনার প্রধান দায়িত্ব:

## মূল নির্দেশনা:
1. **OCR ত্রুটি সংশোধন**: প্রদত্ত প্রসঙ্গে যেকোনো ভুল বানান, বিকৃত অক্ষর বা OCR ত্রুটি দেখলে সেগুলোকে সঠিক বাংলা শব্দে রূপান্তর করুন
2. **নির্ভুল উত্তর**: বিকৃত বা ভুল তথ্য থাকলেও মূল অর্থ বুঝে সঠিক উত্তর দিন
3. **বাংলা ভাষার শুদ্ধতা**: সব সময় পরিশুদ্ধ, ব্যাকরণগতভাবে সঠিক বাংলায় উত্তর দিন


## বিশেষ সতর্কতা:
- যদি প্রসঙ্গে "কলযাণীি" দেখেন, এটি "কল্যাণী" হিসেবে বুঝুন
- যদি "ব্ক" দেখেন, এটি "বিয়ে" বা প্রাসঙ্গিক শব্দ হিসেবে বুঝুন  
- যদি "দেিতাি" দেখেন, এটি "দেবতা" হিসেবে বুঝুন
- যদি "োোলক" দেখেন, এটি "তাহাকে" বা প্রাসঙ্গিক শব্দ হিসেবে বুঝুন
- অস্পষ্ট বা বিকৃত নামের ক্ষেত্রে প্রসঙ্গ অনুযায়ী সবচেয়ে সম্ভাব্য সঠিক নাম ব্যবহার করুন

## উত্তরের মান:
1. **নির্দিষ্ট ও সঠিক**: প্রশ্নের সরাসরি উত্তর দিন, অপ্রয়োজনীয় তথ্য এড়িয়ে চলুন
2. **প্রমাণ-ভিত্তিক**: প্রদত্ত প্রসঙ্গের তথ্যের উপর ভিত্তি করে উত্তর দিন
3. **সুস্পষ্ট বাংলা**: আধুনিক, প্রমিত বাংলায় লিখুন
4. **সংক্ষিপ্ত**: প্রয়োজনীয় তথ্য দিয়ে সংক্ষেপে উত্তর দিন

## যদি তথ্য অপর্যাপ্ত হয়:
"প্রদত্ত প্রসঙ্গে এই প্রশ্নের পূর্ণাঙ্গ উত্তরের জন্য যথেষ্ট তথ্য নেই।"

সর্বদা নির্ভুল, সহায়ক এবং তথ্যনির্ভর উত্তর প্রদান করুন।"""
    
    def health_check(self) -> bool:
        """
        Check if the LLM service is working
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Try a simple generation
            test_response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                model=self.model,
                max_tokens=10,
                temperature=0.1
            )
            
            return test_response.choices[0].message.content is not None
            
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models
        
        Returns:
            List of model names
        """
        try:
            if not self.client:
                return []
            
            models = self.client.models.list()
            return [model.id for model in models.data]
            
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

# Global generator instance
_generator_instance = None

def get_generator(api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
                 temperature: float = 0.5, max_tokens: int = 512) -> GroqGenerator:
    """
    Get or create the global generator instance
    
    Args:
        api_key: Groq API key
        model: Model name
        temperature: Generation temperature
        max_tokens: Maximum tokens
        
    Returns:
        GroqGenerator instance
    """
    global _generator_instance
    
    if _generator_instance is None:
        _generator_instance = GroqGenerator(api_key, model, temperature, max_tokens)
    
    return _generator_instance
