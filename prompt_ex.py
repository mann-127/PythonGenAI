# prompt_engineering_complete.py
import requests
import time
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
 
class GroqLLM:
    """Simplified LLM wrapper with error handling"""
    
    def __init__(self, model="llama-3.3-70b-versatile", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
    
    def invoke(self, prompt: str, max_retries=3) -> str:
        """Robust LLM call with retries and error handling"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 1024
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⏳ Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"API Error {response.status_code}: {response.text}"
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error after {max_retries} attempts: {e}"
                time.sleep(1)
        
        return "Max retries exceeded"
 
@dataclass
class Example:
    input_text: str
    output_text: str
 
class FewShotPrompting:
    """Implementation of Few-Shot Prompting Pattern"""
    
    def __init__(self, llm: GroqLLM):
        self.llm = llm
    
    def sentiment_analysis(self, text: str) -> str:
        """Few-shot sentiment analysis"""
        examples = [
            Example("I love this product! It's amazing.", "Positive"),
            Example("This is the worst service I've ever experienced.", "Negative"),
            Example("The movie was okay, nothing special.", "Neutral"),
            Example("The food was good but the service was slow.", "Mixed")
        ]
        
        prompt = "Classify the sentiment of these texts:\n\n"
        for ex in examples:
            prompt += f"Text: {ex.input_text}\nSentiment: {ex.output_text}\n\n"
        
        prompt += f"Text: {text}\nSentiment:"
        
        return self.llm.invoke(prompt)
    
    def code_generation(self, description: str) -> str:
        """Few-shot code generation"""
        examples = [
            Example(
                "Function to calculate factorial",
                "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)"
            ),
            Example(
                "Function to check palindrome",
                "def is_palindrome(s):\n    return s == s[::-1]"
            ),
            Example(
                "Function to find maximum number in list",
                "def find_max(numbers):\n    return max(numbers)"
            )
        ]
        
        prompt = "Generate Python code based on descriptions:\n\n"
        for ex in examples:
            prompt += f"Description: {ex.input_text}\nCode:\n{ex.output_text}\n\n"
        
        prompt += f"Description: {description}\nCode:"
        
        return self.llm.invoke(prompt)
    
    def text_classification(self, text: str, categories: List[str], examples: List[Example]) -> str:
        """Generic few-shot classification"""
        prompt = f"Classify the text into one of these categories: {', '.join(categories)}\n\n"
        
        for ex in examples:
            prompt += f"Text: {ex.input_text}\nCategory: {ex.output_text}\n\n"
        
        prompt += f"Text: {text}\nCategory:"
        
        return self.llm.invoke(prompt)
 
class ChainOfThoughtPrompting:
    """Implementation of Chain-of-Thought Prompting Pattern"""
    
    def __init__(self, llm: GroqLLM):
        self.llm = llm
    
    def math_problem_solver(self, problem: str) -> str:
        """CoT for math problems"""
        prompt = f"""
    Solve this math problem step by step:
    
    Question: {problem}
    
    Let's think step by step:
    """
        return self.llm.invoke(prompt)
    
    def logic_puzzle_solver(self, puzzle: str) -> str:
        """CoT for logic puzzles"""
        prompt = f"""
    Solve this logic puzzle step by step:
    
    Puzzle: {puzzle}
    
    Let's reason step by step:
    """
        return self.llm.invoke(prompt)
 
    def complex_reasoning(self, scenario: str, domain: str = "general") -> str:
            """CoT for complex reasoning tasks"""
            templates = {
                "medical": "Analyze this medical scenario step by step:\n\n{scenario}\n\nLet's think step by step:",
                "financial": "Analyze this financial scenario step by step:\n\n{scenario}\n\nLet's think step by step:",
                "technical": "Analyze this technical problem step by step:\n\n{scenario}\n\nLet's think step by step:",
                "general": "Analyze this scenario step by step:\n\n{scenario}\n\nLet's think step by step:"
            }
            
            template = templates.get(domain, templates["general"])
            prompt = template.format(scenario=scenario)
            
            return self.llm.invoke(prompt)
        
    def multi_step_cot(self, problem: str, steps: List[str]) -> str:
            """Guided CoT with specific steps"""
            prompt = f"""
    Solve this problem by following these steps:
    
    Problem: {problem}
    
    Steps to follow:
    """
            for i, step in enumerate(steps, 1):
                prompt += f"{i}. {step}\n"
            
            prompt += "\nNow solve step by step:"
            
            return self.llm.invoke(prompt)
 
class RolePrompting:
    """Implementation of Role Prompting Pattern"""
    
    def __init__(self, llm: GroqLLM):
        self.llm = llm
    
    def expert_responder(self, question: str, role: str, expertise: str = "") -> str:
        """Role-based response with specific expertise"""
        prompt = f"""
    You are {role}. {expertise}
    
    Respond to the following question from your professional perspective:
    
    Question: {question}
    
    Response:
    """
        return self.llm.invoke(prompt)
    
    def persona_based_chat(self, message: str, persona: str, history: List[Tuple[str, str]] = None) -> str:
        """Persona-based conversation"""
        prompt = f"""
    Adopt this persona: {persona}
    
    """
        if history:
            prompt += "Previous conversation:\n"
            for user_msg, assistant_msg in history[-3:]:
                prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
        
        prompt += f"User: {message}\nAssistant:"
        
        return self.llm.invoke(prompt)
    
    def tone_controlled_response(self, message: str, tone: str, context: str = "") -> str:
        """Response with controlled tone"""
        tones = {
            "formal": "Respond in a formal, professional tone.",
            "casual": "Respond in a casual, friendly tone.",
            "empathetic": "Respond with empathy and understanding.",
            "technical": "Respond with technical precision and detail.",
            "simple": "Respond in simple, easy-to-understand language."
        }
        
        tone_instruction = tones.get(tone, "Respond appropriately.")
        
        prompt = f"""
    Context: {context}
    Tone: {tone_instruction}
    
    Message: {message}
    
    Response:
    """
        return self.llm.invoke(prompt)
 
class CombinedPromptEngine:
    """Combines all prompting patterns for advanced applications"""
    
    def __init__(self, llm: GroqLLM):
        self.llm = llm
        self.few_shot = FewShotPrompting(llm)
        self.chain_of_thought = ChainOfThoughtPrompting(llm)
        self.role_prompting = RolePrompting(llm)
    
    def advanced_reasoning(self, problem: str, role: str, examples: List[Example] = None, require_cot: bool = True) -> str:
        """Combine role + few-shot + CoT"""
        prompt_parts = []
        
        # Role assignment
        prompt_parts.append(f"You are {role}.")
        
        # Few-shot examples
        if examples:
            prompt_parts.append("Learn from these examples:")
            for i, ex in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Input: {ex.input_text}")
                prompt_parts.append(f"Output: {ex.output_text}")
            prompt_parts.append("")
        
        # Problem statement
        prompt_parts.append(f"Problem: {problem}")
        
        # Chain-of-thought instruction
        if require_cot:
            prompt_parts.append("Please reason step by step before providing your final answer.")
        
        return self.llm.invoke("\n\n".join(prompt_parts))
    
    def educational_tutor(self, question: str, subject: str, student_level: str = "beginner") -> str:
        """Educational tutor combining all patterns"""
        examples = [
            Example(
                "What is photosynthesis?",
                "Photosynthesis is how plants make food using sunlight. Let me explain step by step:\n1. Plants take in sunlight\n2. They absorb water and carbon dioxide\n3. Using chlorophyll, they convert these into glucose (sugar)\n4. This process releases oxygen\nSo plants essentially eat sunlight!"
            )
        ]
        
        role = f"a {subject} tutor for {student_level} students. Explain concepts clearly with examples and encouragement."
        
        return self.advanced_reasoning(question, role, examples, require_cot=True)
    
    def customer_service_agent(self, complaint: str, company_context: str = "") -> str:
        """Customer service agent with combined patterns"""
        examples = [
            Example(
                "My order is late",
                "I understand your frustration about the delayed order. Let me help you:\n1. First, I'll check your order status\n2. If there's a delay, I'll investigate the cause\n3. I'll provide solutions like expedited shipping or discounts\nPlease share your order number so I can assist you better."
            ),
            Example(
                "The product is damaged",
                "I'm sorry to hear the product arrived damaged. Here's what we'll do:\n1. I'll arrange for a replacement immediately\n2. You'll receive a return label for the damaged item\n3. We'll expedite the replacement shipping\nPlease share photos of the damage and your order details."
            )
        ]
        
        role = f"a professional customer service agent for {company_context}. Be empathetic, solution-oriented, and clear."
        
        return self.advanced_reasoning(complaint, role, examples, require_cot=False)
 
class PracticalApplications:
    """Real-world applications using prompt patterns"""
    
    def __init__(self, llm: GroqLLM):
        self.engine = CombinedPromptEngine(llm)
    
    def math_tutor_app(self):
        """Interactive math tutor application"""
        print("🧮 Math Tutor Application")
        print("Type 'quit' to exit\n")
        
        while True:
            problem = input("Enter math problem: ").strip()
            if problem.lower() == 'quit':
                break
            
            response = self.engine.educational_tutor(problem, "mathematics")
            print(f"\n📚 Tutor: {response}\n")
    
    def customer_service_bot(self, company: str = "TechStore"):
        """Customer service chatbot"""
        print(f"🎯 {company} Customer Service")
        print("How can I help you today? Type 'quit' to exit\n")
        
        conversation_history = []
        
        while True:
            message = input("Customer: ").strip()
            if message.lower() == 'quit':
                break
            
            response = self.engine.customer_service_agent(message, company)
            print(f"\n🤖 Agent: {response}\n")
            
            conversation_history.append((message, response))
 
    def code_review_assistant(self):
            """Code review and explanation assistant"""
            print("💻 Code Review Assistant")
            print("Paste your code or describe what you need help with. Type 'quit' to exit\n")
            
            while True:
                code_input = input("Your code/query: ").strip()
                if code_input.lower() == 'quit':
                    break
                
                role = "a senior software engineer with 10+ years of experience. Provide clear, constructive code reviews and explanations."
                
                response = self.engine.advanced_reasoning(
                    problem=f"Review/explain this: {code_input}",
                    role=role,
                    require_cot=True
                )
                print(f"\n👨‍💻 Senior Engineer: {response}\n")
 
class PromptEvaluator:
    """Evaluate prompt effectiveness"""
    
    def __init__(self, llm: GroqLLM):
        self.llm = llm
    
    def evaluate_response(self, prompt: str, response: str, criteria: List[str]) -> str:
        """Evaluate response quality"""
        criteria_text = "\n".join([f"- {c}" for c in criteria])
        
        evaluation_prompt = f"""
    Evaluate this AI response based on the following criteria:
    {criteria_text}
    
    Original Prompt: {prompt}
    AI Response: {response}
    
    Provide scores 1-5 for each criterion and overall feedback:
    """
        return self.llm.invoke(evaluation_prompt)
    
    def compare_prompt_patterns(self, problem: str, patterns: Dict[str, str]):
        """Compare different prompt patterns on the same problem"""
        results = {}
        
        for pattern_name, prompt in patterns.items():
            print(f"Testing {pattern_name}...")
            response = self.llm.invoke(prompt)
            results[pattern_name] = {
                'prompt': prompt,
                'response': response,
                'length': len(response)
            }
            time.sleep(1)  # Rate limit protection
        
        return results


def demo_all_patterns():
    """Comprehensive demonstration of all prompt patterns"""
    llm = GroqLLM()
    
    print("🚀 COMPLETE PROMPT ENGINEERING DEMO")
    print("=" * 50)
    
    # Initialize all pattern classes
    few_shot = FewShotPrompting(llm)
    cot = ChainOfThoughtPrompting(llm)
    role = RolePrompting(llm)
    combined = CombinedPromptEngine(llm)
    apps = PracticalApplications(llm)
    evaluator = PromptEvaluator(llm)
    
    # Demo 1: Few-Shot Prompting
    print("\n1. 🎯 FEW-SHOT PROMPTING")
    print("-" * 30)
    
    sentiment_text = "The battery life is incredible but the screen is too small"
    sentiment = few_shot.sentiment_analysis(sentiment_text)
    print(f"Text: {sentiment_text}")
    print(f"Sentiment: {sentiment}")
    
    code_desc = "Function to calculate fibonacci sequence"
    code = few_shot.code_generation(code_desc)
    print(f"\nCode Generation for: {code_desc}")
    print(f"Code: {code}")
    
    # Demo 2: Chain-of-Thought
    print("\n2. 🧠 CHAIN-OF-THOUGHT PROMPTING")
    print("-" * 30)
    
    math_problem = "If a car travels 60 mph for 2 hours, then 75 mph for 1 hour, what's the average speed for the entire trip?"
    math_solution = cot.math_problem_solver(math_problem)
    print(f"Math Problem: {math_problem}")
    print(f"Solution: {math_solution}")
    
    logic_puzzle = "There are three boxes: one contains apples, one contains oranges, and one contains both. All labels are wrong. You can pick one fruit from one box. How do you determine the contents of all boxes?"
    logic_solution = cot.logic_puzzle_solver(logic_puzzle)
    print(f"\nLogic Puzzle: {logic_puzzle}")
    print(f"Solution: {logic_solution}")
    
    # Demo 3: Role Prompting
    print("\n3. 🎭 ROLE PROMPTING")
    print("-" * 30)
    
    medical_question = "What are the benefits of regular exercise?"
    medical_response = role.expert_responder(medical_question, "a certified doctor", "Provide evidence-based medical advice.")
    print(f"Medical Question: {medical_question}")
    print(f"Doctor's Response: {medical_response}")
    
    tech_question = "Should I learn Python or JavaScript first?"
    tech_response = role.expert_responder(tech_question, "a senior software engineer", "Give practical career advice.")
    print(f"\nTech Question: {tech_question}")
    print(f"Engineer's Response: {tech_response}")
    
    # Demo 4: Combined Patterns
    print("\n4. 🔥 COMBINED PATTERNS")
    print("-" * 30)
    
    complex_problem = "A startup has $100,000 funding. They need to allocate budget for marketing (40%), development (35%), and operations (25%). Marketing returns 150% ROI, development returns 200% ROI, operations returns 50% ROI. What's the optimal allocation?"
    finance_advisor = "a financial consultant with MBA from Harvard. Provide detailed analysis with calculations."
    
    finance_examples = [
        Example(
            "A company has $50K to allocate between two projects with different ROIs",
            "Let's calculate: Project A ROI 120%, Project B ROI 80%. Optimal allocation depends on risk tolerance. Generally, allocate more to higher ROI projects while maintaining operational needs."
        )
    ]
    
    combined_response = combined.advanced_reasoning(complex_problem, finance_advisor, finance_examples, True)
    print(f"Business Problem: {complex_problem}")
    print(f"Consultant's Analysis: {combined_response}")
    
    # Demo 5: Real Applications
    print("\n5. 🛠️ REAL-WORLD APPLICATIONS")
    print("-" * 30)
    
    # Quick test of applications
    test_complaint = "I ordered a laptop 5 days ago and it hasn't shipped yet. This is unacceptable!"
    service_response = apps.engine.customer_service_agent(test_complaint, "TechStore")
    print(f"Customer Complaint: {test_complaint}")
    print(f"Service Response: {service_response}")
    
    # Demo 6: Evaluation
    print("\n6. 📊 PROMPT EVALUATION")
    print("-" * 30)
    
    criteria = [
        "Accuracy of information",
        "Clarity of explanation",
        "Completeness of response",
        "Appropriateness for audience",
        "Practical usefulness"
    ]
    
    evaluation = evaluator.evaluate_response(
        "Explain quantum computing to a 10-year-old",
        "Quantum computing is like having a magical computer that can be 0 and 1 at the same time!",
        criteria
    )
    print(f"Evaluation Results:\n{evaluation}")
    
    return {
        'few_shot': few_shot,
        'chain_of_thought': cot,
        'role_prompting': role,
        'combined_engine': combined,
        'applications': apps,
        'evaluator': evaluator
    }
 
def interactive_demo():
    """Interactive demo for users to try different patterns"""
    # llm = GroqLLM()
    tools = demo_all_patterns()
    # application = PracticalApplications()
    
    print("\n" + "=" * 50)
    print("🎮 INTERACTIVE DEMO MODE")
    print("=" * 50)
    
    while True:
        print("\nChoose a demo:")
        print("1. 🧮 Math Tutor")
        print("2. 🎯 Customer Service Bot")
        print("3. 💻 Code Review Assistant")
        print("4. 🧠 Custom Chain-of-Thought")
        print("5. 🎭 Custom Role Playing")
        print("6. 📊 Evaluate a Response")
        print("7. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            tools['applications'].math_tutor_app() # application.math_tutor_app()
        elif choice == '2':
            company = input("Enter company name (or press enter for default): ").strip()
            if not company:
                company = "TechCorp"
            tools['applications'].customer_service_bot(company)
        elif choice == '3':
            tools['applications'].code_review_assistant()
        elif choice == '4':
            problem = input("Enter your problem: ").strip()
            response = tools['chain_of_thought'].complex_reasoning(problem)
            print(f"\n🤔 Chain-of-Thought Analysis:\n{response}")
        elif choice == '5':
            role = input("Enter role (e.g., 'history professor', 'fitness coach'): ").strip()
            question = input("Enter your question: ").strip()
            response = tools['role_prompting'].expert_responder(question, role)
            print(f"\n🎭 {role} Response:\n{response}")
        elif choice == '6':
            prompt = input("Enter the original prompt: ").strip()
            response = input("Enter the AI response to evaluate: ").strip()
            criteria = ["Accuracy", "Clarity", "Helpfulness", "Completeness"]
            evaluation = tools['evaluator'].evaluate_response(prompt, response, criteria)
            print(f"\n📊 Evaluation:\n{evaluation}")
        elif choice == '7':
            print("👋 Thanks for exploring Prompt Engineering!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
 
if __name__ == "__main__":
    try:
        # Run comprehensive demo
        demo_tools = demo_all_patterns()
        
        # Start interactive mode
        interactive_demo()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have a stable internet connection and valid API key.")
