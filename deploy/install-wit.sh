#!/usr/bin/env bash
# Install the `wit` launcher to ~/.local/bin so you can run `wit` from any directory.
set -euo pipefail
mkdir -p "$HOME/.local/bin"
install -m 0755 "$(dirname "$0")/wit" "$HOME/.local/bin/wit"
echo "installed wit -> $HOME/.local/bin/wit"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) : ;;
  *) echo "note: add ~/.local/bin to your PATH (e.g. in ~/.bashrc)";;
esac
echo "run it: cd <your repo> && wit"
