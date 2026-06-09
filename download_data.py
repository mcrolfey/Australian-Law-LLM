import time
import kagglehub
import requests

print("Starting robust dataset download...")
max_retries = 5
retry_delay = 10

for attempt in range(max_retries):
    try:
        # This will attempt to download or resume the download
        path = kagglehub.dataset_download("umarbutler/open-australian-legal-corpus")
        print("\nSUCCESS! Dataset fully downloaded to:", path)
        break
    except (requests.exceptions.RequestException, Exception) as e:
        print(f"\n[Warning] Connection dropped on attempt {attempt + 1}/{max_retries}: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("\n[Error] Failed after multiple retries. Please check your internet connection or try a stable Wi-Fi/Ethernet line.")