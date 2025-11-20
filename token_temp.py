import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
 
# ============================================
# Example 1: Data Classification (Low Temperature)
# ============================================
 
def classify_product(product_name, temperature=0.1):
    """
    Use LOW temperature for consistent classification
    We want the same category every time for the same product
    """
    
    prompt = f"Classify '{product_name}' into ONE category: Electronics, Clothing, Food, or Books. Reply with only the category name."
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,  # LOW = consistent
        "max_tokens": 10  # SHORT = just need category name
    }
    
    print(f"\n--- Classification (temp={temperature}) ---")
    print(f"Product: {product_name}")
    
    # Simulate multiple calls to show consistency
    for i in range(3):
        # response = make_api_call(payload)
        print(f"Call {i+1}: Electronics")  # With temp=0.1, always same result
    
    return "Electronics"
 
 
# ============================================
# Example 2: Creative Marketing (High Temperature)
# ============================================
 
def generate_marketing_copy(product_name, temperature=0.9):
    """
    Use HIGH temperature for creative, varied marketing copy
    We want different catchy phrases each time
    """
    
    prompt = f"Write a catchy marketing tagline for {product_name}. Be creative and engaging!"
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,  # HIGH = creative variety
        "max_tokens": 30  # MEDIUM = enough for tagline
    }
    
    print(f"\n--- Marketing Copy (temp={temperature}) ---")
    print(f"Product: {product_name}")
    
    # Show variety in responses
    sample_responses = [
        "Unleash Your Digital Potential!",
        "Where Performance Meets Elegance",
        "Power That Moves With You"
    ]
    
    for i, response in enumerate(sample_responses, 1):
        print(f"Call {i}: {response}")
    
    return sample_responses[0]
 
 
# ============================================
# Example 3: Detailed Analysis (High Max Tokens)
# ============================================
 
def detailed_product_analysis(product_name, max_tokens=500):
    """
    Use HIGH max_tokens for detailed analysis
    We need comprehensive information
    """
    
    prompt = f"Provide a detailed analysis of {product_name} including features, target market, pricing strategy, and competitive advantages."
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,  # BALANCED
        "max_tokens": max_tokens  # HIGH = detailed response
    }
    
    print(f"\n--- Detailed Analysis (max_tokens={max_tokens}) ---")
    print(f"Product: {product_name}")
    print("Response length: ~375 words")
    print("Includes: Features, Market, Pricing, Competition...")
    
    return "Detailed analysis..."
 
 
# ============================================
# Example 4: Quick Summaries (Low Max Tokens)
# ============================================
 
def quick_summary(product_description, max_tokens=20):
    """
    Use LOW max_tokens for brief summaries
    We just need key points
    """
    
    prompt = f"Summarize in 5 words or less: {product_description}"
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,  # BALANCED
        "max_tokens": max_tokens  # LOW = brief response
    }
    
    print(f"\n--- Quick Summary (max_tokens={max_tokens}) ---")
    print(f"Description: {product_description}")
    print(f"Summary: Premium wireless gaming mouse")
    
    return "Premium wireless gaming mouse"
 
 
 
# ============================================
# COMPARISON: Different Configurations
# ============================================
 
def compare_configurations():
    """
    Show how different temp/token settings affect output
    """
    
    product = "Gaming Laptop"
    
    print("="*60)
    print("CONFIGURATION COMPARISON")
    print("="*60)
    
    # Config 1: Strict Classification
    print("\n🎯 Config 1: STRICT (temp=0.0, tokens=10)")
    print("Use: Data extraction, categorization")
    print("Result: Always 'Electronics' - Consistent & Predictable")
    
    # Config 2: Balanced General Purpose
    print("\n⚖️ Config 2: BALANCED (temp=0.7, tokens=200)")
    print("Use: Product descriptions, general tasks")
    print("Result: 'High-performance gaming laptop with RTX graphics...'")
    print("Length: ~150 words, varied but reasonable")
    
    # Config 3: Creative Marketing
    print("\n🎨 Config 3: CREATIVE (temp=1.2, tokens=50)")
    print("Use: Marketing, brainstorming")
    print("Result: 'Dominate Every Game, Anywhere You Go! 🎮'")
    print("Length: Short, punchy, very creative")
    
    # Config 4: Detailed Analysis
    print("\n📊 Config 4: ANALYTICAL (temp=0.3, tokens=1000)")
    print("Use: Reports, detailed analysis")
    print("Result: Comprehensive 750-word analysis with specs...")
    print("Length: Very detailed, factual, consistent")
 
 
# ============================================
# COST & PERFORMANCE CONSIDERATIONS
# ============================================
 
def explain_tradeoffs():
    """
    Explain practical implications
    """
    
    print("\n" + "="*60)
    print("💰 COST & PERFORMANCE IMPLICATIONS")
    print("="*60)
    
    print("\n1. API COSTS (tokens = money)")
    print("   - max_tokens=50   → ~$0.0001 per call")
    print("   - max_tokens=500  → ~$0.001 per call")
    print("   - max_tokens=2000 → ~$0.004 per call")
    print("   💡 Use lower tokens when possible!")
    
    print("\n2. RESPONSE SPEED")
    print("   - max_tokens=50   → 0.5 seconds")
    print("   - max_tokens=500  → 2-3 seconds")
    print("   - max_tokens=2000 → 5-8 seconds")
    print("   ⚡ Lower tokens = faster responses")
    
    print("\n3. TEMPERATURE EFFECTS")
    print("   - temp=0.0 → Cacheable, consistent")
    print("   - temp=0.7 → Good balance")
    print("   - temp=1.5 → Unpredictable, harder to validate")
    print("   🎯 Lower temp = more reliable for production")
 
 
# ============================================
# ETL-SPECIFIC RECOMMENDATIONS
# ============================================
 
def etl_recommendations():
    """
    Best practices for ETL pipelines
    """
    
    print("\n" + "="*60)
    print("🔧 ETL PIPELINE RECOMMENDATIONS")
    print("="*60)
    
    configs = {
        "Product Categorization": {
            "temperature": 0.1,
            "max_tokens": 20,
            "reason": "Need consistent categories"
        },
        "Feature Extraction": {
            "temperature": 0.3,
            "max_tokens": 100,
            "reason": "Factual, brief, reliable"
        },
        "Marketing Tags": {
            "temperature": 0.7,
            "max_tokens": 50,
            "reason": "Creative but controlled"
        },
        "Product Descriptions": {
            "temperature": 0.5,
            "max_tokens": 200,
            "reason": "Detailed but not excessive"
        },
        "Sentiment Analysis": {
            "temperature": 0.0,
            "max_tokens": 10,
            "reason": "Binary/categorical output"
        }
    }
    
    for task, config in configs.items():
        print(f"\n📌 {task}:")
        print(f"   Temperature: {config['temperature']}")
        print(f"   Max Tokens:  {config['max_tokens']}")
        print(f"   Why: {config['reason']}")
 
# ============================================
# Run Examples
# ============================================
 
if __name__ == "__main__":
    # Run comparisons
    classify_product("MacBook Pro")
    generate_marketing_copy("Wireless Mouse")
    detailed_product_analysis("Gaming Headset")
    quick_summary("Professional noise-cancelling over-ear headphones with 30-hour battery")
    
    # Show comparisons
    compare_configurations()
    
    # Show tradeoffs
    explain_tradeoffs()
    
    # ETL-specific recommendations
    etl_recommendations()
    
    print("\n" + "="*60)
    print("💡 KEY TAKEAWAY")
    print("="*60)
    print("Temperature = Creativity dial (0=boring, 2=wild)")
    print("Max Tokens  = Length limit (10=tweet, 2000=essay)")
    print("\nFor ETL: Use LOW temp + LOW tokens = Fast + Cheap + Reliable")
    print("="*60)
