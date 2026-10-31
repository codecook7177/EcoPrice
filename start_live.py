"""
EcoPrice - One-Click Live Cloud Launcher
Starts the Flask ML server and establishes a secure public HTTPS Cloudflare Tunnel.
Displays the live URL and automatically launches the browser.
"""

import sys
import time
import subprocess
import re
import webbrowser
import socket
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def is_port_in_use(port: int = 5000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def drain_stream(stream, log_file):
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            for line in iter(stream.readline, ""):
                f.write(line)
                f.flush()
                if "ERR" in line or "error" in line.lower() or "502" in line:
                    print(f"   [Tunnel] {line.strip()}", flush=True)
    except Exception:
        pass

def main():
    print("=" * 70)
    print("   ECOPRICE - A REAL-TIME DYNAMIC PRICING & SPOILAGE PREDICTION ALGORITHM")
    print("=" * 70)

    flask_proc = None
    if is_port_in_use(5000):
        print("\n[1/3] EcoPrice Flask app is already active on http://127.0.0.1:5000")
    else:
        print("\n[1/3] Starting Flask Application & ML Inference Engine...")
        flask_proc = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=str(BASE_DIR)
        )
        # Actively poll until Flask ML service is fully loaded and listening
        print("      Loading machine learning models and dataset", end="", flush=True)
        ready = False
        for _ in range(40):
            if is_port_in_use(5000):
                ready = True
                break
            time.sleep(0.5)
            print(".", end="", flush=True)
            if flask_proc.poll() is not None:
                print("\n[ERROR] Flask server process exited prematurely with code", flask_proc.returncode)
                return

        if not ready:
            print("\n[ERROR] Flask failed to start listening on port 5000 within 20s.")
            if flask_proc:
                flask_proc.terminate()
            return
        print(" Ready!")

    print("[2/3] Initializing Secure Cloudflare Public Edge Tunnel...")

    cloudflared_bin = BASE_DIR / "cloudflared.exe"
    if not cloudflared_bin.exists():
        print(f"[ERROR] cloudflared.exe not found at {cloudflared_bin}")
        print("Please ensure cloudflared.exe is present in the project root.")
        if flask_proc:
            flask_proc.terminate()
        return

    # Start cloudflared with resilient auto-protocol negotiation
    cf_proc = subprocess.Popen(
        [str(cloudflared_bin), "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:5000"],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    url = None
    print("[3/3] Acquiring Public HTTPS URL...", flush=True)

    start_t = time.time()
    for line in iter(cf_proc.stdout.readline, ""):
        match = re.search(r"https://(?!api\b)[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            break
        if time.time() - start_t > 30:
            break

    # Spawn thread to continuously monitor and drain pipe to log file
    tunnel_log = BASE_DIR / "data" / "tunnel.log"
    threading.Thread(target=drain_stream, args=(cf_proc.stdout, str(tunnel_log)), daemon=True).start()

    local_ip = get_local_ip()

    if url:
        try:
            (BASE_DIR / "data" / "live_url.txt").write_text(url.strip())
        except Exception:
            pass
        print("\n" + "=" * 70, flush=True)
        print("   SUCCESS! ECOPRICE IS LIVE ONLINE", flush=True)
        print("=" * 70, flush=True)
        print(f"\n   1. PUBLIC LIVE URL : {url}", flush=True)
        if local_ip != "127.0.0.1":
            print(f"   2. WI-FI LAN URL   : http://{local_ip}:5000  <-- (Use this if college Wi-Fi blocks Cloudflare!)", flush=True)
        print(f"   3. LOCAL PC URL    : http://127.0.0.1:5000\n", flush=True)
        print("   Keep this terminal window OPEN during your demo.", flush=True)
        print("   Press CTRL+C anytime in this window to stop the server.", flush=True)
        print("=" * 70 + "\n", flush=True)
        sys.stdout.flush()

        try:
            webbrowser.open(url)
        except Exception:
            pass

        try:
            while True:
                if flask_proc and flask_proc.poll() is not None:
                    print("\n[!] Flask server process terminated unexpectedly.")
                    break
                if cf_proc.poll() is not None:
                    print("\n[!] Cloudflared tunnel process terminated unexpectedly.")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down EcoPrice servers...")
            cf_proc.terminate()
            if flask_proc:
                flask_proc.terminate()
            print("Done. Goodbye!")
    else:
        print("\n[!] Could not automatically capture the tunnel URL within 30s.")
        print("Please check your internet connection or use local URL: http://127.0.0.1:5000")
        try:
            while True:
                if flask_proc and flask_proc.poll() is not None:
                    break
                if cf_proc.poll() is not None:
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            cf_proc.terminate()
            if flask_proc:
                flask_proc.terminate()

if __name__ == "__main__":
    main()
