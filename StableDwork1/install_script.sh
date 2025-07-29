#!/bin/bash

# Stable Diffusion CLI System Installer for macOS
# This script installs and sets up the SD CLI system

set -e

echo "🚀 Installing Stable Diffusion CLI System..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This installer is designed for macOS"
    exit 1
fi

# Check for Python 3.8+
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    echo "Install Python 3.8+ using Homebrew: brew install python@3.10"
    exit 1
fi

# Create installation directory
INSTALL_DIR="$HOME/.sd_cli"
mkdir -p "$INSTALL_DIR"

# Download the main script (assuming it's saved as sd_manager.py)
echo "📥 Downloading SD Manager..."
# In practice, you'd download from a repository or provide the file
cp sd_manager.py "$INSTALL_DIR/sd_manager.py"

# Make executable
chmod +x "$INSTALL_DIR/sd_manager.py"

# Create symlink for global access
SYMLINK_PATH="/usr/local/bin/sd"
if [[ -L "$SYMLINK_PATH" ]]; then
    rm "$SYMLINK_PATH"
fi

# Create wrapper script
cat > "$INSTALL_DIR/sd" << 'EOF'
#!/bin/bash
exec python3 "$HOME/.sd_cli/sd_manager.py" "$@"
EOF

chmod +x "$INSTALL_DIR/sd"

# Try to create symlink (may need sudo)
if ln -sf "$INSTALL_DIR/sd" "$SYMLINK_PATH" 2>/dev/null; then
    echo "✅ Global 'sd' command installed"
else
    echo "⚠️  Could not create global symlink. Run manually:"
    echo "sudo ln -sf '$INSTALL_DIR/sd' '$SYMLINK_PATH'"
    echo ""
    echo "Or add to your PATH:"
    echo "echo 'export PATH=\$PATH:$INSTALL_DIR' >> ~/.zshrc"
    echo "source ~/.zshrc"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run 'sd setup' to install dependencies"
echo "2. Run 'sd dataset prepare --path /path/to/images --name my_dataset'"
echo "3. Run 'sd train --dataset my_dataset --model my_model'"
echo "4. Run 'sd generate --prompt \"your prompt here\" --model my_model'"
echo ""
echo "📖 For help: sd --help"
