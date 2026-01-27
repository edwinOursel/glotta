#!/usr/bin/env python3
"""
Simple HTTP server to serve the Flutter web build.

Usage:
    python serve_web.py [port]

Default port: 8080

This allows you to test the web app locally, especially useful in Termux
where you can't run `flutter run -d chrome` directly.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# Default port
PORT = 8080

if len(sys.argv) > 1:
    try:
        PORT = int(sys.argv[1])
    except ValueError:
        print(f"Invalid port: {sys.argv[1]}")
        sys.exit(1)

# Web build directory
WEB_DIR = Path(__file__).parent / "build" / "web"

if not WEB_DIR.exists():
    print("❌ Web build not found!")
    print(f"   Looking for: {WEB_DIR}")
    print()
    print("Please build the web app first:")
    print("   1. On a machine with Flutter installed:")
    print("      flutter build web")
    print()
    print("   2. Transfer the build/web/ folder to mobile/build/web/")
    print()
    print("   3. Or use the pre-built version if available")
    sys.exit(1)

# Change to web directory
os.chdir(WEB_DIR)

# Custom handler to set correct MIME types
class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Set CORS headers for API calls
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def guess_type(self, path):
        # Ensure correct MIME types for Flutter files
        mime_type, _ = super().guess_type(path)
        if path.endswith('.js'):
            return 'application/javascript'
        if path.endswith('.wasm'):
            return 'application/wasm'
        return mime_type

# Create server
Handler = MyHTTPRequestHandler

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"""
╔════════════════════════════════════════════════════════════╗
║              Glotta Web App Server                         ║
╚════════════════════════════════════════════════════════════╝

🌐 Server running at:
   http://localhost:{PORT}

📱 On this device, open your browser to:
   http://localhost:{PORT}

🔗 From other devices on the same network:
   http://<your-ip>:{PORT}

💡 Tips:
   - Make sure the backend API is running (core/api_server.py)
   - Update API base URL in the app settings if needed
   - Press Ctrl+C to stop the server

Serving from: {WEB_DIR}
        """)

        httpd.serve_forever()

except KeyboardInterrupt:
    print("\n\n👋 Server stopped. Bye!")
except OSError as e:
    if e.errno == 98:  # Address already in use
        print(f"\n❌ Port {PORT} is already in use!")
        print(f"   Try a different port: python serve_web.py 8081")
    else:
        raise
