#!/usr/bin/env python3
"""
Download the latest Flutter web build from GitHub Actions artifacts.

This script allows you to easily download pre-built web app on Termux or
any environment without Flutter SDK.

Prerequisites:
    pip install requests
    # or with uv: uv pip install -e .

Usage:
    python download_build.py [--token YOUR_GITHUB_TOKEN]

The script will:
1. Find the latest successful build
2. Download the artifact
3. Extract it to mobile/build/web/
4. You can then run: python serve_web.py
"""

import os
import sys
import zipfile
from pathlib import Path
import json
import argparse

try:
    import requests
except ImportError:
    print("❌ Error: 'requests' module not found!")
    print()
    print("Please install it with:")
    print("   uv pip install -e .     # If using uv")
    print("   pip install requests    # Or with pip")
    print()
    sys.exit(1)

# GitHub repository info (update with your repo)
REPO_OWNER = "edwinOursel"
REPO_NAME = "glotta"
ARTIFACT_NAME = "flutter-web-build"

# Paths
SCRIPT_DIR = Path(__file__).parent
BUILD_DIR = SCRIPT_DIR / "build" / "web"


def get_latest_artifact_url(token=None):
    """Get the download URL for the latest artifact."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Get workflow runs
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs"
    params = {
        "status": "success",
        "per_page": 10,
    }

    print("🔍 Searching for latest successful build...")
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"❌ Failed to get workflow runs: {response.status_code}")
        if response.status_code == 401:
            print("   Authentication failed. Try providing a GitHub token:")
            print("   python download_build.py --token YOUR_TOKEN")
        print(f"   Response: {response.text}")
        return None

    runs = response.json().get("workflow_runs", [])

    if not runs:
        print("❌ No successful builds found")
        return None

    # Find the latest run with our artifact
    for run in runs:
        run_id = run["id"]
        run_number = run["run_number"]
        created_at = run["created_at"]

        # Get artifacts for this run
        artifacts_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{run_id}/artifacts"
        artifacts_response = requests.get(artifacts_url, headers=headers)

        if artifacts_response.status_code != 200:
            continue

        artifacts = artifacts_response.json().get("artifacts", [])

        for artifact in artifacts:
            if artifact["name"] == ARTIFACT_NAME:
                print(f"✓ Found build #{run_number} from {created_at}")
                return artifact["archive_download_url"], headers, run_number

    print("❌ No artifacts found with name:", ARTIFACT_NAME)
    return None


def download_artifact(url, headers):
    """Download the artifact zip file."""
    print("📥 Downloading artifact...")

    response = requests.get(url, headers=headers, stream=True)

    if response.status_code != 200:
        print(f"❌ Failed to download artifact: {response.status_code}")
        return None

    # Save to temporary file
    zip_path = SCRIPT_DIR / "flutter-web-build.zip"

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"   {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')

    print(f"\n✓ Downloaded to: {zip_path}")
    return zip_path


def extract_artifact(zip_path):
    """Extract the artifact to build/web/."""
    print(f"📦 Extracting to: {BUILD_DIR}")

    # Create build directory
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BUILD_DIR)

    # Remove zip file
    zip_path.unlink()

    # Show build info if available
    build_info_path = BUILD_DIR / "build-info.txt"
    if build_info_path.exists():
        print("\n📋 Build info:")
        with open(build_info_path, 'r') as f:
            for line in f:
                print(f"   {line.rstrip()}")

    print(f"\n✅ Build extracted successfully!")
    print(f"   Location: {BUILD_DIR}")
    print(f"\n🚀 To run the app:")
    print(f"   python serve_web.py")
    print(f"   Then open: http://localhost:8080")


def main():
    parser = argparse.ArgumentParser(
        description="Download Flutter web build from GitHub Actions"
    )
    parser.add_argument(
        "--token",
        help="GitHub personal access token (optional, but recommended for private repos)",
        default=os.environ.get("GITHUB_TOKEN")
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if build directory exists"
    )

    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════════════╗
║         Glotta - Download Web Build                       ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Check if build already exists
    if BUILD_DIR.exists() and not args.force:
        print(f"⚠️  Build directory already exists: {BUILD_DIR}")
        response = input("   Download anyway? (y/N): ")
        if response.lower() != 'y':
            print("   Cancelled.")
            return

        # Clean up existing build
        import shutil
        shutil.rmtree(BUILD_DIR)

    # Get artifact URL
    result = get_latest_artifact_url(args.token)

    if not result:
        print("\n💡 Tips:")
        print("   - Ensure at least one successful build exists")
        print("   - Check: https://github.com/{}/{}/actions".format(REPO_OWNER, REPO_NAME))
        if not args.token:
            print("   - For private repos, provide a token:")
            print("     python download_build.py --token YOUR_TOKEN")
            print("   - Create token at: https://github.com/settings/tokens")
            print("     (needs 'repo' or 'actions:read' scope)")
        sys.exit(1)

    download_url, headers, run_number = result

    # Download
    zip_path = download_artifact(download_url, headers)

    if not zip_path:
        sys.exit(1)

    # Extract
    extract_artifact(zip_path)

    print("\n✨ Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
