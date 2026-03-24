# MarketScope Backend

FastAPI-powered backend for the **MarketScope** Indian Stock Market Heatmap tool.
Fetches real-time intraday data from Yahoo Finance for 15+ sectors covering 250+ NSE-listed stocks.

---

## Setup

```bash
# 1. Clone / download this folder
git clone <your-repo-url>
cd marketscope-backend

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env   # Windows: copy .env.example .env

# 5. Start the server
python main.py
```

The API will be available at **http://localhost:8000**.

---

## Environment Variables (`.env`)

| Variable                  | Default | Description                                    |
|---------------------------|---------|------------------------------------------------|
| `PORT`                    | `8000`  | Port the server listens on                     |
| `CACHE_DURATION_SECONDS`  | `300`   | Seconds before cached data is considered stale |
| `MARKET_OPEN_TIME`        | `09:15` | NSE market open time (informational)           |
| `MARKET_CLOSE_TIME`       | `15:30` | NSE market close time (informational)          |
| `TELEGRAM_BOT_TOKEN`      | ``      | Telegram bot token for Trade Guardian alerts   |
| `TELEGRAM_CHAT_ID`        | ``      | Telegram chat id for alert delivery            |
| `TRADE_GUARDIAN_POLL_SECONDS` | `10` | Trade Guardian monitoring interval in seconds |
| `TRADE_GUARDIAN_STARTUP_DELAY_SECONDS` | `20` | Delay before Trade Guardian starts polling after app boot |
| `TRADE_GUARDIAN_REPEAT_SECONDS` | `60` | Repeat interval for unacknowledged alerts  |
| `FO_RADAR_MAX_SYMBOLS` | `36` | Max F&O Radar symbols to refresh in each automatic background cycle |
| `FO_RADAR_MIN_REFRESH_SECONDS` | `900` | Minimum seconds between automatic F&O Radar refreshes |

---

## API Endpoints

### `GET /`
Welcome endpoint — returns app name and available routes.

**Response:**
```json
{
  "app": "MarketScope API",
  "endpoints": ["/heatmap", "/health"],
  "description": "Indian Stock Market Heatmap Backend"
}
```

---

### `GET /heatmap`
Full sector and stock heatmap data, sorted by absolute change %.

**Response:**
```json
{
  "sectors": [
    {
      "name": "PHARMA",
      "change_pct": -1.24,
      "stocks": [
        {
          "symbol": "DRREDDY",
          "ltp": 1240.50,
          "change_pct": -2.10,
          "volume_ratio": 1.8,
          "fo": true
        }
      ]
    }
  ],
  "last_updated": "14:35:22",
  "market_open": true
}
```

| Field          | Type    | Description                                           |
|----------------|---------|-------------------------------------------------------|
| `symbol`       | string  | NSE ticker (without `.NS`)                            |
| `ltp`          | float   | Last traded price                                     |
| `change_pct`   | float   | Intraday % change from day open                       |
| `volume_ratio` | float   | Current candle volume ÷ average candle volume         |
| `fo`           | bool    | Whether the stock is F&O eligible                     |

---

### `GET /health`
Server and cache health check.

**Response:**
```json
{
  "status": "ok",
  "last_updated": "14:35:22",
  "market_open": true,
  "total_sectors": 16,
  "total_stocks": 253
}
```

---

## Trade Guardian

Trade Guardian is a manual trade tracking layer that monitors live prices and sends Telegram alerts for:

- entry triggered
- stop loss hit
- target 1 hit
- target 2 hit
- repeated reminders for unacknowledged critical alerts

### Trade Guardian endpoints

- `GET /api/trade-guardian`
- `GET /api/trade-guardian/trades`
- `GET /api/trade-guardian/trades/{trade_id}`
- `POST /api/trade-guardian/trades`
- `POST /api/trade-guardian/trades/{trade_id}/close`
- `GET /api/trade-guardian/alerts`
- `POST /api/trade-guardian/alerts/{alert_id}/acknowledge`
- `POST /api/trade-guardian/monitor`
- `POST /api/trade-guardian/test-telegram`

### Trade Guardian create payload

Required fields:

- `symbol`
- `direction`
- `entry_price`
- `stop_loss`
- `target_1`
- `target_2`

Optional fields:

- `quantity`
- `notes`

---

## Connect React Frontend

Replace mock data in your React app with a live fetch:

```js
useEffect(() => {
  fetch('http://localhost:8000/heatmap')
    .then(r => r.json())
    .then(d => setData(d));
}, []);
```

For production, replace `http://localhost:8000` with your deployed backend URL.

---

## How It Works

1. **Startup** — On server start, an initial full fetch runs immediately to warm the cache.
2. **Scheduler** — APScheduler triggers a background refresh every **5 minutes**, but only during NSE market hours (Mon–Fri, 09:00–15:35 IST).
3. **On-demand refresh** — If a `/heatmap` request arrives with stale cache during market hours, a synchronous refresh is performed before responding.
4. **yfinance batching** — Each sector's symbols are downloaded in a single `yf.download()` batch call, with a 0.2 s sleep between sectors to avoid rate limiting.

---

## Deploy to Railway

1. Push this folder to a GitHub repository.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Select your repository.
4. Add environment variables from `.env.example` in the Railway dashboard.
5. Railway auto-detects Python and deploys automatically.
6. Your live URL will be: `https://your-app.up.railway.app`

> **Tip:** Set a `Procfile` or `railway.toml` start command if needed:
> ```
> web: uvicorn main:app --host 0.0.0.0 --port $PORT
> ```

---

## Project Structure

```
marketscope-backend/
├── main.py          # FastAPI app, routes, lifespan
├── fetcher.py       # yfinance data download & processing
├── scheduler.py     # APScheduler 5-min background job
├── cache.py         # In-memory cache with staleness tracking
├── stocks.py        # Sector → symbol mappings + F&O set
├── requirements.txt # Python dependencies
├── .env.example     # Environment variable template
├── .gitignore       # Git ignore rules
└── README.md        # This file
```

---

## License

MIT
