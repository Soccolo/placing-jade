# Notrix

A bridge between portfolio recipes and brokerage accounts. Notrix connects to your Alpaca paper trading account and displays your account status alongside a target portfolio allocation.

**⚠️ PAPER TRADING ONLY** - This application is designed exclusively for Alpaca paper trading accounts.

## What Notrix Does

- **Reads a portfolio recipe** from a local CSV file
- **Connects to Alpaca** paper trading accounts securely
- **Displays account info**: cash, buying power, portfolio value, positions
- **Shows target allocations** from the CSV file with validation

Notrix does NOT:
- Generate rebalancing plans
- Place trades automatically
- Modify your positions

## Financial Command Center

Notrix now includes a Financial Command Center with dedicated tabs for:
- Live Market Ticker
- Stock Research & Analysis (charts, indicators, predictions, news & sentiment)
- Portfolio Calculator (Daily Profitability Calculator)
- Economic News
- Insurance News
- Insurance Industry Tracker

Live data sources for the Financial Command Center are configured in code.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- An Alpaca paper trading account ([Sign up here](https://alpaca.markets/))

### Installation

1. **Clone the repository**
   ```bash
   cd notrix-app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate an encryption key**
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```
   Copy the output key.

5. **Create the `.env` file**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and paste your encryption key:
   ```
   ENCRYPTION_KEY=your-generated-key-here
   ```

6. **Run the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Open your browser**
   Navigate to: http://localhost:8000

## Streamlit App (Community Cloud)

Run locally:
```bash
streamlit run streamlit_app.py
```

Deploy on Streamlit Community Cloud:
1. Push this repo to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Set the entrypoint to `streamlit_app.py`.
4. Add `ENCRYPTION_KEY` in Streamlit Secrets.

Note: Streamlit Community Cloud uses ephemeral storage; stored credentials may reset on redeploy.

## Deploy on Fly.io (Recommended)

Fly.io is a good fit for the FastAPI app and supports persistent volumes for SQLite.

1. **Install Fly CLI**  
   https://fly.io/docs/flyctl/install/

2. **Login**
   ```bash
   fly auth login
   ```

3. **Create the app (choose a unique name)**
   ```bash
   fly launch --name notrix-app --no-deploy
   ```
   Update the `app` value in `fly.toml` if you picked a different name.

4. **Create a volume for SQLite**
   ```bash
   fly volumes create notrix_data --size 1
   ```

5. **Set required secrets**
   ```bash
   fly secrets set ENCRYPTION_KEY=your-generated-key-here
   ```

6. **Deploy**
   ```bash
   fly deploy
   ```

## Usage

### 1. Connect Your Alpaca Account

1. Go to the **Connect** page
2. Enter your Alpaca **paper trading** API key and secret
3. Click **Connect**
4. Your credentials are encrypted and stored securely

Get your paper trading API keys from: https://app.alpaca.markets/paper/dashboard/overview

### 2. View Dashboard

Once connected, the **Dashboard** shows:
- Portfolio value, equity, cash, and buying power
- All current positions with P/L information
- A manual **Refresh** button to update data

### 3. View Target Portfolio

The **Strategy** page displays:
- All symbols and weights from your target portfolio CSV
- Validation status (valid/invalid)
- Any errors or warnings
- A **Reload from Disk** button to refresh if you edit the CSV

## Target Portfolio CSV Format

The portfolio file is located at `data/target_portfolio.csv`.

**Required columns:**
- `symbol` - Stock ticker symbol (e.g., AAPL, MSFT)
- `weight` - Decimal weight between 0 and 1 (e.g., 0.25 for 25%)

**Example:**
```csv
symbol,weight
AAPL,0.30
MSFT,0.20
SPY,0.50
```

**Validation rules:**
- No duplicate symbols
- No negative weights
- Sum of weights must equal 1.0 (±0.01 tolerance)

## Project Structure

```
notrix-app/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration from environment
│   ├── database.py          # SQLite database models
│   ├── routes/
│   │   ├── connect.py       # Credential management endpoints
│   │   ├── dashboard.py     # Account display endpoints
│   │   └── strategy.py      # Portfolio display endpoints
│   ├── services/
│   │   ├── alpaca.py        # Alpaca API integration
│   │   ├── credentials.py   # Credential storage
│   │   ├── encryption.py    # Encryption utilities
│   │   └── portfolio.py     # CSV loading and validation
│   ├── static/
│   │   └── style.css        # Application styles
│   └── templates/
│       ├── base.html        # Base template
│       ├── connect.html     # Connect page
│       ├── dashboard.html   # Dashboard page
│       └── strategy.html    # Strategy page
├── data/
│   └── target_portfolio.csv # Your target portfolio
├── tests/
│   └── test_portfolio.py    # Portfolio validation tests
├── requirements.txt
├── .env.example
└── README.md
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_portfolio.py -v
```

## Configuration

All configuration is done through environment variables (set in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `ENCRYPTION_KEY` | Fernet encryption key (required) | - |
| `DATABASE_URL` | SQLite database path | `sqlite+aiosqlite:///./notrix.db` |
| `TARGET_PORTFOLIO_PATH` | Path to portfolio CSV | `data/target_portfolio.csv` |
| `WEIGHT_SUM_EPSILON` | Weight sum tolerance | `0.01` |

## Security

- **Credentials are encrypted** at rest using Fernet symmetric encryption
- **Secrets are never logged** - only connection status is recorded
- **Paper trading only** - the application is configured for Alpaca paper trading
- **Audit logging** - connection and refresh events are logged to the database

## Development

Run in development mode with auto-reload:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## License

MIT
