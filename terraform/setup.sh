#!/bin/bash

# Mettre à jour les paquets
sudo apt update -y
sudo apt upgrade -y

# Installer Docker
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
# curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
# echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update -y
sudo apt install -y docker-ce
# sudo systemctl start docker
# sudo systemctl enable docker

# Installer Python 3.11 et venv
sudo apt install -y python3.11 python3.11-venv

# Installer Tesseract
sudo apt install -y tesseract-ocr

# Installer Git
sudo apt install -y git

# Cloner le projet
cd /home/debian
git clone https://github.com/MathisPRS/Bot-Discord-Trie.git
cd Bot-Discord-Trie

# Créer un environnement virtuel et installer les dépendances
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Installer les dépendances de Playwright
playwright install-deps
playwright install

# Créer un service systemd pour lancer bot.py
sudo tee /etc/systemd/system/bot-discord.service > /dev/null <<EOF
[Unit]
Description=Bot Discord Service
After=network.target

[Service]
User=debian
Group=debian
WorkingDirectory=/home/debian/Bot-Discord-Trie
ExecStartPre=/bin/bash -c 'echo $$ > /home/debian/Bot-Discord-Trie/run/bot-ia.pid'
ExecStart=/home/debian/Bot-Discord-Trie/venv/bin/python /home/debian/Bot-Discord-Trie/bot-ia.py
ExecStopPost=/bin/bash -c 'rm -f /home/debian/Bot-Discord-Trie/run/bot-ia.pid'
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Recharger systemd, démarrer et activer le service
sudo systemctl daemon-reload
