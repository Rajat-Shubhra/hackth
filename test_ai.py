import json
import os

# Test Google AI API
def test_google_ai():
    try:
        # Load settings
        with open('fraudguard_config.json', 'r') as f:
            settings = json.load(f)
        
        api_key = settings.get('google_api_key', '')
        if not api_key:
            print("❌ No API key found in settings")
            return False
        
        print(f"✅ API key found: {api_key[:10]}...")
        
        # Test Google AI
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model_name = settings.get('working_model', 'gemini-1.5-flash')
        print(f"🤖 Testing model: {model_name}")
        
        model = genai.GenerativeModel(model_name)
        
        # Simple test
        test_prompt = "Analyze this transaction for fraud risk: Amount: $1500, Type: Online Purchase, Location: New York. Give a brief risk assessment."
        
        print("🔄 Sending test request...")
        response = model.generate_content(test_prompt)
        
        if response and response.text:
            print("✅ AI Response received!")
            print("📝 Response preview:", response.text[:200] + "...")
            return True
        else:
            print("❌ No response from AI")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Google AI: {e}")
        return False

if __name__ == "__main__":
    test_google_ai()
