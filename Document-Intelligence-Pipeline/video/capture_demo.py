"""
Takes the screenshot for the demo slide from the real app: starts Streamlit,
uploads a CUAD contract, clicks Extract (one Gemini call) and captures the
result. Uses the system's Chrome through Playwright -- no browser download.

    python video/capture_demo.py [contract id, default c015]
"""

import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "build" / "demo.png"
PORT = 8599


def main(contract_id: str = "c015") -> None:
    pdf = ROOT / "data" / "contracts" / f"{contract_id}.pdf"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
         "--server.headless", "true", "--browser.gatherUsageStats", "false",
         # Hide the developer toolbar (Stop / Deploy) from the screenshot.
         "--client.toolbarMode", "viewer"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://localhost:{PORT}"
        for _ in range(60):  # wait for the server to come up
            try:
                if requests.get(f"{url}/_stcore/health", timeout=1).ok:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.5)

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            page = browser.new_page(viewport={"width": 1600, "height": 960}, device_scale_factor=1)
            page.goto(url)
            page.locator("input[type=file]").set_input_files(pdf)
            page.get_by_role("button", name="Extract").click()
            # The metrics row appears once the extraction has finished; the
            # tables are canvas-drawn and paint a moment after that.
            page.get_by_text("Validation issues").wait_for(timeout=120_000)
            page.locator("[data-testid=stDataFrame] canvas").first.wait_for(timeout=30_000)
            page.get_by_text("Running...").wait_for(state="hidden", timeout=30_000)
            time.sleep(3)
            page.screenshot(path=OUT)
            browser.close()
        print(f"saved {OUT}")
    finally:
        server.terminate()


if __name__ == "__main__":
    main(*sys.argv[1:])
