# Notrix Deployment Guide

This guide covers deploying Notrix to the web using multiple platforms.

---

## Option 1: Fly.io (Recommended - Free Tier Available)

### Prerequisites
- Install flyctl: `curl -L https://fly.io/install.sh | sh`
- Create account: `flyctl auth signup` or `flyctl auth login`

### Step-by-Step Deployment

#### 1. Generate Encryption Key
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
Copy the output (you'll need it in step 4).

#### 2. Initialize Fly.io App
```bash
cd /home/vladsft/notrix-app
flyctl launch --no-deploy
```
- Choose your app name (or keep default)
- Select region (closest to you)
- Say **NO** to Postgres database
- Say **NO** to Redis

#### 3. Create Persistent Storage
```bash
flyctl volumes create notrix_data --region iad --size 1
```
(Replace `iad` with your chosen region)

#### 4. Set Environment Variables
```bash
# Set your encryption key (use the one you generated in step 1)
flyctl secrets set ENCRYPTION_KEY="your-generated-key-here"

# These are already in fly.toml, but you can override if needed:
# flyctl secrets set DATABASE_URL="sqlite+aiosqlite:////data/notrix.db"
# flyctl secrets set TARGET_PORTFOLIO_PATH="/app/data/target_portfolio.csv"
```

#### 5. Deploy
```bash
flyctl deploy
```

#### 6. Access Your App
```bash
flyctl open
```
Or visit: `https://your-app-name.fly.dev`

#### Manage Your App
```bash
# View logs
flyctl logs

# SSH into your app
flyctl ssh console

# Check status
flyctl status

# Scale down to free tier (if needed)
flyctl scale count 1

# Update secrets
flyctl secrets set ENCRYPTION_KEY="new-key"
```

---

## Option 2: Railway (Free Tier Available)

### Step-by-Step Deployment

#### 1. Install Railway CLI
```bash
npm install -g @railway/cli
# OR
curl -fsSL https://railway.app/install.sh | sh
```

#### 2. Login and Initialize
```bash
cd /home/vladsft/notrix-app
railway login
railway init
```

#### 3. Generate Encryption Key
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

#### 4. Set Environment Variables
```bash
railway variables set ENCRYPTION_KEY="your-generated-key-here"
railway variables set DATABASE_URL="sqlite+aiosqlite:///./notrix.db"
railway variables set TARGET_PORTFOLIO_PATH="data/target_portfolio.csv"
railway variables set WEIGHT_SUM_EPSILON="0.01"
```

#### 5. Create railway.json
Create a file named `railway.json`:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

#### 6. Deploy
```bash
railway up
```

#### 7. Get Your URL
```bash
railway domain
```

---

## Option 3: Render (Free Tier Available)

### Step-by-Step Deployment

#### 1. Push to GitHub
```bash
cd /home/vladsft/notrix-app
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/notrix-app.git
git push -u origin main
```

#### 2. Create Render Account
- Go to https://render.com
- Sign up with GitHub

#### 3. Create New Web Service
- Click "New" → "Web Service"
- Connect your GitHub repository
- Configure:
  - **Name**: notrix-app
  - **Environment**: Python 3
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### 4. Add Environment Variables
In Render dashboard, add:
- `ENCRYPTION_KEY`: (generate with Python command above)
- `DATABASE_URL`: `sqlite+aiosqlite:///./notrix.db`
- `TARGET_PORTFOLIO_PATH`: `data/target_portfolio.csv`
- `WEIGHT_SUM_EPSILON`: `0.01`
- `PYTHON_VERSION`: `3.12.0`

#### 5. Add Disk Storage
- In your service settings, go to "Disks"
- Add disk:
  - Name: `notrix-data`
  - Mount Path: `/data`
  - Size: 1 GB

#### 6. Deploy
Click "Create Web Service" - Render will automatically deploy.

Your app will be at: `https://notrix-app.onrender.com`

---

## Option 4: DigitalOcean App Platform

### Step-by-Step Deployment

#### 1. Push to GitHub
```bash
cd /home/vladsft/notrix-app
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/notrix-app.git
git push -u origin main
```

#### 2. Create DigitalOcean App
- Go to https://cloud.digitalocean.com/apps
- Click "Create App"
- Connect GitHub repository

#### 3. Configure App
- **Resource Type**: Web Service
- **Dockerfile Path**: Dockerfile
- **HTTP Port**: 8000
- **Instance Size**: Basic ($5/month)

#### 4. Add Environment Variables
- `ENCRYPTION_KEY`: (generate with Python command)
- `DATABASE_URL`: `sqlite+aiosqlite:///./notrix.db`
- `TARGET_PORTFOLIO_PATH`: `data/target_portfolio.csv`
- `WEIGHT_SUM_EPSILON`: `0.01`

#### 5. Deploy
Click "Create Resources"

---

## Important Notes for All Platforms

### Database Persistence
- **SQLite** works for single-instance deployments
- For multiple instances, consider upgrading to PostgreSQL:
  ```bash
  # Update DATABASE_URL to:
  DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/notrix"
  
  # Install additional dependency:
  pip install asyncpg
  ```

### Security Checklist
✅ Always set a strong `ENCRYPTION_KEY`  
✅ Never commit `.env` file to git  
✅ Use HTTPS (most platforms auto-enable)  
✅ Review API key permissions in Alpaca  
✅ Only use paper trading credentials  

### Custom Domain
Most platforms support custom domains:
- **Fly.io**: `flyctl certs add yourdomain.com`
- **Railway**: Add in dashboard → Settings → Domains
- **Render**: Add in dashboard → Settings → Custom Domain

### Monitoring
```bash
# Fly.io
flyctl logs --app notrix-app

# Railway
railway logs

# Render
Check dashboard → Logs tab
```

---

## Testing Your Deployment

1. Visit your deployed URL
2. Go to `/connect` page
3. Enter your Alpaca paper trading credentials
4. Verify connection works
5. Check `/dashboard` displays account info
6. Check `/strategy` shows your portfolio

---

## Troubleshooting

### "Application failed to start"
- Check logs for Python errors
- Verify `ENCRYPTION_KEY` is set
- Ensure all dependencies in `requirements.txt`

### "500 Internal Server Error"
- Check encryption key is valid base64
- Verify database path is writable
- Check logs for specific error

### Database not persisting
- Ensure persistent volume is mounted
- Check DATABASE_URL points to mounted path
- Verify write permissions

### Can't connect to Alpaca
- Verify you're using **paper trading** credentials
- Check API keys are correct (no extra spaces)
- Test credentials locally first

---

## Cost Comparison

| Platform | Free Tier | Paid (Basic) |
|----------|-----------|--------------|
| **Fly.io** | 3 VMs, 3GB storage | ~$5/month |
| **Railway** | $5 credit/month | $5/month + usage |
| **Render** | 750 hours/month | $7/month |
| **DigitalOcean** | $200 credit (60 days) | $5/month |

**Recommendation**: Start with Fly.io free tier for testing, upgrade to paid if you need 24/7 uptime.
