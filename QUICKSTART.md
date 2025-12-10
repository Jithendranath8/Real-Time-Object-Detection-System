# Quick Start Guide

## ⚡ Fast Setup (3 Steps)

1. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate  # macOS/Linux
   # OR
   venv\Scripts\activate     # Windows
   ```

2. **Verify setup (optional but recommended):**
   ```bash
   python check_setup.py
   ```

3. **Run the app:**
   ```bash
   streamlit run src/app_streamlit.py
   ```

## 🔧 Common Issues & Quick Fixes

### "ModuleNotFoundError: No module named 'ultralytics'"
**Fix:** Your virtual environment is not activated!
```bash
# Check if venv is active (you should see (venv) in prompt)
source venv/bin/activate  # Activate it
pip install -r requirements.txt  # Reinstall if needed
```

### "ModuleNotFoundError: No module named 'src'"
**Fix:** This is now automatically handled. If you still see it:
- Make sure you're in the project root directory
- Run: `streamlit run src/app_streamlit.py` (not from inside src/)

### App won't start
**Fix:** Run the setup checker:
```bash
python check_setup.py
```
This will tell you exactly what's missing.

## 📝 Full Installation (First Time)

If you haven't set up the project yet:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python check_setup.py

# 5. Run
streamlit run src/app_streamlit.py
```

## ✅ Verification Checklist

Before running the app, make sure:
- [ ] Virtual environment is activated (see `(venv)` in terminal)
- [ ] All dependencies installed (`python check_setup.py` passes)
- [ ] You're in the project root directory
- [ ] Python version is 3.10+ (`python --version`)

