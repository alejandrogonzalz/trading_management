import requests
import json

API_BASE = "http://localhost:8001"

def test_lead_history():
    print("--- Testing /lead/history (GLOBAL) ---")
    # Test without symbol to verify global fetch
    url = f"{API_BASE}/lead/history"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Total Records: {len(data)}")
            if len(data) > 0:
                print("First record sample:")
                # Check for critical fields the UI needs
                sample = data[0]
                print(json.dumps(sample, indent=2))
                
                # Validation logic
                has_price = "price" in sample or "avgPrice" in sample
                has_symbol = "symbol" in sample
                has_time = "time" in sample or "updateTime" in sample
                
                print(f"Validation -> Has Price: {has_price}, Has Symbol: {has_symbol}, Has Time: {has_time}")
            else:
                print("❌ No history records found in DB.")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

def test_lead_open_orders():
    print("\n--- Testing /lead/open-orders ---")
    url = f"{API_BASE}/lead/open-orders"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Total Open Orders: {len(data)}")
            if len(data) > 0:
                print("Order sample:")
                print(json.dumps(data[0], indent=2))
            else:
                print("No open orders active currently.")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_lead_history()
    test_lead_open_orders()
