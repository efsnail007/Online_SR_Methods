# Online_SR_Methods

## Запуск по локальной сети

Сценарий: бэкенд работает на этом ПК, фронтенд запускается на ноутбуке с камерой.

### 1. Узнать IP этого ПК

На ПК с бэкендом в PowerShell:

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -ne "127.0.0.1" -and $_.IPAddress -notlike "169.254*" } |
  Select-Object IPAddress, InterfaceAlias
```

Дальше в примерах используется `192.168.1.10`; замените его на IP этого ПК.

### 2. Запустить бэкенд на ПК

```powershell
docker compose up backend
```

Бэкенд слушает `0.0.0.0:8000`, поэтому он доступен с другого устройства в той же Wi-Fi/LAN-сети:

```text
http://192.168.1.10:8000/api/v1/health
```

Если ноутбук не видит бэкенд, разрешите входящие подключения TCP `8000` в Windows Firewall.

### 3. Запустить фронтенд на ноутбуке

На ноутбуке:

```powershell
cd src/frontend
npm install
$env:VITE_API_BASE_URL="http://192.168.1.10:8000/api/v1"
npm run dev -- --host 0.0.0.0
```

Откройте в браузере ноутбука:

```text
http://localhost:5173
```

Камера в браузере надежнее всего работает именно через `localhost`. Если адрес API нужно поменять без перезапуска фронта, откройте:

```text
http://localhost:5173/?api=http://192.168.1.10:8000/api/v1
```

Или измените поле `Backend URL` в интерфейсе; значение сохранится в браузере ноутбука.
