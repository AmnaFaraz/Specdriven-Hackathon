import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from services.gemini_service import GeminiService

def test_gemini_service():
    """
    Test the Gemini service to ensure it's working correctly with the API key.
    """
    print("Testing Gemini Service...")
    
    try:
        # Initialize the Gemini service
        gemini_service = GeminiService()
        print("[OK] Gemini service initialized successfully")

        # Test basic generation
        test_prompt = "Hello, how are you? Just respond with a simple greeting."
        response = gemini_service.generate_response(test_prompt)

        if response and len(response) > 0:
            print(f"[OK] Gemini API call successful")
            print(f"Response: {response[:100]}...")  # Print first 100 chars
        else:
            print("[ERROR] Gemini API call failed - no response")
            return False

        # Test RAG functionality (without actual Qdrant data)
        rag_response = gemini_service.query_rag(
            query="What is robotics?",
            context="entire_book",
            selected_text=None,
            user_id=None
        )

        print(f"[OK] RAG query processed")
        print(f"RAG Response: {rag_response['response'][:100]}...")  # Print first 100 chars

        print("\n[OK] All tests passed! The Gemini API integration is working correctly.")
        return True

    except Exception as e:
        print(f"[ERROR] Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemini_service()
    if success:
        print("\n[SUCCESS] Gemini API integration is successfully implemented!")
    else:
        print("\n[FAILURE] There were issues with the Gemini API integration.")