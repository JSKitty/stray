#!/bin/sh
# Stray installer — https://stray.jskitty.cat
# Usage: curl -sSf https://stray.jskitty.cat/install.sh | sh
set -e

REPO="JSKitty/stray"
INSTALL_DIR="${STRAY_INSTALL_DIR:-$HOME/.local/bin}"

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

case "$OS" in
    Darwin) os="macos" ;;
    Linux)  os="linux" ;;
    *)
        echo "Error: Unsupported OS: $OS"
        exit 1
        ;;
esac

case "$ARCH" in
    x86_64|amd64)  arch="x86_64" ;;
    aarch64|arm64) arch="aarch64" ;;
    *)
        echo "Error: Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

BINARY="stray-${os}-${arch}"

# Get latest release tag
echo "Fetching latest release..."
LATEST=$(curl -sSf "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')

if [ -z "$LATEST" ]; then
    echo "Error: Could not determine latest release"
    exit 1
fi

URL="https://github.com/${REPO}/releases/download/${LATEST}/${BINARY}.tar.gz"

echo "Installing stray ${LATEST} (${os}/${arch})..."

# Download and extract
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if ! curl -sSfL "$URL" -o "$TMP/stray.tar.gz"; then
    echo "Error: Download failed — no release binary for your platform?"
    echo "Try building from source: cargo install --git https://github.com/${REPO}"
    exit 1
fi

tar xzf "$TMP/stray.tar.gz" -C "$TMP"

# Install
mkdir -p "$INSTALL_DIR"
mv "$TMP/stray" "$INSTALL_DIR/stray"
chmod +x "$INSTALL_DIR/stray"

echo ""
echo "  /\\_/\\    Stray ${LATEST} installed!"
echo " ( o.o )   Location: ${INSTALL_DIR}/stray"
echo "  > ^ <"
echo ""

# Check if install dir is in PATH
case ":$PATH:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
        echo "Note: ${INSTALL_DIR} is not in your PATH."
        echo "Add it: export PATH=\"${INSTALL_DIR}:\$PATH\""
        echo ""
        ;;
esac

echo "Run 'stray' to start."
