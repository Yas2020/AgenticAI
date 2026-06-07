#!/bin/bash

# Install for Standard Python (GIL)
echo "-------------------------------------------"
echo "--- Installing Standard Python Packages ---"
echo "-------------------------------------------"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt


echo "--- Setting up oh-my-zsh ---"

# Define the custom plugins directory
ZSH_CUSTOM=${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}

# Clone plugins if they don't exist
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-autosuggestions" ]; then
    git clone https://github.com/zsh-users/zsh-autosuggestions "$ZSH_CUSTOM/plugins/zsh-autosuggestions"
fi

if [ ! -d "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting" ]; then
    git clone https://github.com/zsh-users/zsh-syntax-highlighting "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting"
fi