import time
import requests

URL = "https://jewelify-server.onrender.com/"

def call_url():
    while True:
        try:
            response = requests.get(URL)
            print(f"Called {URL}, Status Code: {response.status_code}")
        except Exception as e:
            print(f"Error calling {URL}: {e}")
        
        time.sleep(14 * 60)  # Wait for 14 minutes

if __name__ == "__main__":
    call_url()
