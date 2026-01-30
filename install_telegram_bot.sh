#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/opt/telegram_bot"
SERVICE_USER="telegrambot"
SERVICE_GROUP="telegrambot"
BOT_FILE="telegram_image_bot.py"
VENV_DIR="$WORKDIR/.venv"
ENV_FILE="/etc/telegram_bot.env"
SYSTEMD_UNIT="/etc/systemd/system/telegram_bot.service"

echo "=== Установка Telegram Image Bot ==="

if [ "$(id -u)" -ne 0 ]; then
  echo "Этот скрипт должен быть запущен с правами root (sudo)." >&2
  exit 1
fi

# Проверяем, что бот-файл существует
if [ ! -f "./$BOT_FILE" ]; then
  echo "Файл $BOT_FILE не найден в $(pwd)." >&2
  exit 1
fi

# Установка Python и зависимостей
echo "1) Проверка Python..."
apt update -y
apt install -y python3 python3-venv python3-pip

# Определяем реальный интерпретатор Python
PYTHON_BIN=$(command -v python3)
PYTHON_VER=$($PYTHON_BIN -V)
echo "Используем $PYTHON_BIN ($PYTHON_VER)"

# Создаём пользователя, если не существует
if id -u "$SERVICE_USER" >/dev/null 2>&1; then
  echo "Пользователь $SERVICE_USER уже существует."
else
  echo "2) Создание системного пользователя $SERVICE_USER..."
  adduser --system --group --no-create-home "$SERVICE_USER"
fi

# Копируем бота
echo "3) Копирование файлов в $WORKDIR..."
mkdir -p "$WORKDIR"
cp -f "./$BOT_FILE" "$WORKDIR/"
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$WORKDIR"
chmod 750 "$WORKDIR"
chmod 640 "$WORKDIR/$BOT_FILE"

# Создание виртуального окружения и установка зависимостей
echo "4) Создание виртуального окружения..."
$PYTHON_BIN -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install python-telegram-bot aiohttp

# Настройка переменных окружения
echo "5) Настройка переменных окружения..."
read -rp "Введите TELEGRAM_TOKEN: " TELEGRAM_TOKEN
read -rp "Введите OPENAI_API_KEY: " OPENAI_API_KEY
read -rp "Введите ADMIN_TELEGRAM_ID (число): " ADMIN_TELEGRAM_ID
read -rp "Введите MODEL (Enter для gpt-4o-mini): " MODEL
MODEL=${MODEL:-gpt-4o-mini}

cat > "$ENV_FILE" <<EOF
TELEGRAM_TOKEN=$TELEGRAM_TOKEN
OPENAI_API_KEY=$OPENAI_API_KEY
ADMIN_TELEGRAM_ID=$ADMIN_TELEGRAM_ID
MODEL=$MODEL
BOT_DB=$WORKDIR/bot_users.db
EOF

chmod 600 "$ENV_FILE"
chown root:root "$ENV_FILE"

# Создаём systemd unit
echo "6) Создание systemd unit..."
cat > "$SYSTEMD_UNIT" <<EOF
[Unit]
Description=Telegram OpenAI Image Bot
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$WORKDIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_DIR/bin/python $WORKDIR/$BOT_FILE
Restart=always
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

chmod 644 "$SYSTEMD_UNIT"

# Активация и запуск
echo "7) Активация и запуск сервиса..."
systemctl daemon-reload
systemctl enable telegram_bot.service
systemctl restart telegram_bot.service

echo
echo "=== Установка завершена успешно ==="
echo "Проверить статус: sudo systemctl status telegram_bot.service"
echo "Логи в реальном времени: sudo journalctl -u telegram_bot.service -f"
echo
