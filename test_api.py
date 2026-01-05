"""
Script untuk test API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint"""
    print("=" * 60)
    print("Testing Health Check")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_predict():
    """Test predict endpoint"""
    print("=" * 60)
    print("Testing Predict Endpoint")
    print("=" * 60)
    
    test_cases = [
        "Daun padi menguning dan muncul bercak coklat",
        "Apa itu penyakit blas?",
        "Batang padi berlubang dan anakan mati",
        "Bagaimana gejala wereng coklat?",
    ]
    
    for text in test_cases:
        print(f"\nInput: {text}")
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Entities found:")
            for entity in data["entities"]:
                print(f"  {entity['entity_type']}: {entity['entities']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        print("-" * 60)


def test_batch_predict():
    """Test batch predict endpoint"""
    print("=" * 60)
    print("Testing Batch Predict Endpoint")
    print("=" * 60)
    
    texts = [
        "Daun padi menguning",
        "Apa itu penyakit blas?",
        "Batang padi berlubang",
        "Bagaimana gejala wereng coklat?",
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"texts": texts}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total results: {len(data['results'])}")
        for i, result in enumerate(data["results"], 1):
            print(f"\n[{i}] {result['text']}")
            if result["entities"]:
                for entity in result["entities"]:
                    print(f"  {entity['entity_type']}: {entity['entities']}")
            else:
                print("  No entities found")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    print()


def test_extract_entities():
    """Test extract entities endpoint"""
    print("=" * 60)
    print("Testing Extract Entities Endpoint")
    print("=" * 60)
    
    text = "Daun padi menguning dan muncul bercak coklat"
    print(f"Input: {text}")
    
    response = requests.post(
        f"{BASE_URL}/extract-entities",
        json={"text": text}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Entities: {json.dumps(data['entities'], indent=2, ensure_ascii=False)}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    print()


def test_qa_reasoning():
    """Test QA dengan reasoning endpoint"""
    print("=" * 60)
    print("Testing QA with Reasoning Endpoint")
    print("=" * 60)
    
    test_questions = [
        "Daun padi menguning dan muncul bercak coklat",
        "Apa itu penyakit blas?",
        "Batang padi berlubang dan anakan mati",
        "Bagaimana gejala wereng coklat?",
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = requests.post(
            f"{BASE_URL}/qa",
            json={"text": question}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Intent: {data.get('intent')}")
            print(f"Entities: {data.get('entities')}")
            if data.get('reasoning'):
                reasoning = data['reasoning']
                print(f"Reasoning Type: {reasoning.get('type')}")
                if 'results' in reasoning:
                    print(f"Results found: {len(reasoning['results'])}")
                    for i, result in enumerate(reasoning['results'][:3], 1):  # Show first 3
                        if isinstance(result, dict):
                            print(f"  {i}. {result.get('nama', 'N/A')} (score: {result.get('skor', 'N/A')})")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        print("-" * 60)


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("API TESTING")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print("Make sure the API server is running!")
    print()
    
    try:
        # Test health check
        test_health()
        
        # Test predict
        test_predict()
        
        # Test batch predict
        test_batch_predict()
        
        # Test extract entities
        test_extract_entities()
        
        # Test QA with reasoning
        test_qa_reasoning()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API server!")
        print("   Make sure the server is running at", BASE_URL)
        print("   Run: python run_api.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

