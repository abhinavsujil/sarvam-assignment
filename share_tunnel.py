import time
import secrets
from gradio.networking import setup_tunnel

print("[*] Connecting public Gradio tunnel to http://127.0.0.1:7860...")
share_token = secrets.token_hex(16)
share_url = setup_tunnel("127.0.0.1", 7860, share_token, None, None)
print("\n" + "=" * 65)
print(f"[*] PUBLIC GRADIO SHARE URL: {share_url}")
print("[*] Share this exact URL with Shivam to test the voice desk!")
print("=" * 65 + "\n", flush=True)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("[*] Tunnel closed.")
