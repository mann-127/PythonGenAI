"""
Simple Groq LangChain - Step by Step from Main
"""
 
import requests
import time
import os
from dotenv import load_dotenv
 
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
 
def call_groq_api(prompt: str) -> str:
    """Make a single API call to Groq"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Exception: {e}"
 
def step1_extract_topic(text: str) -> str:
    """Step 1: Extract main topic"""
    print("🔹 STEP 1: Extracting topic...")
    prompt = f"What is the main topic of this text? Return only the topic: {text}"
    result = call_groq_api(prompt)
    print(f"   Topic: {result}")
    return result
 
def step2_generate_summary(topic: str) -> str:
    """Step 2: Generate summary"""
    print("🔹 STEP 2: Generating summary...")
    prompt = f"Provide a 2-sentence summary about: {topic}"
    result = call_groq_api(prompt)
    print(f"   Summary: {result}")
    return result
 
def step3_create_questions(summary: str) -> str:
    """Step 3: Create questions"""
    print("🔹 STEP 3: Creating questions...")
    prompt = f"Generate 2 interesting questions about: {summary}"
    result = call_groq_api(prompt)
    print(f"   Questions: {result}")
    return result
 
def main():
    """Main method - starting point"""
    print("🚀 Groq LangChain - Step by Step")
    print("=" * 50)
    
    # Input text
    # input_text = "Machine learning algorithms can analyze medical images to detect diseases early with high accuracy, helping doctors make better diagnoses."
    input_text = "Supercars are cool, because they are fast and quick as compared to the normal cars, but they are also expensive and have high maintanence costs."

    print(f"📥 Input: {input_text}")
    print("\n" + "🔗 Starting Chain Execution..." + "\n")
    
    # Step 1: Extract topic
    topic = step1_extract_topic(input_text)
    time.sleep(1)  # Rate limit protection
    
    # Step 2: Generate summary
    summary = step2_generate_summary(topic)
    time.sleep(1)
    
    # Step 3: Create questions
    questions = step3_create_questions(summary)
    
    # Final result
    print("\n" + "=" * 50)
    print("🎉 FINAL OUTPUT:")
    print("=" * 50)
    print(f"Topic: {topic}")
    print(f"Summary: {summary}")
    print(f"Questions: {questions}")
    
    print("\n📚 Chain completed successfully!")
 
if __name__ == "__main__":
    main()
