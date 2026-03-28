# Running the Streamlit UI

## Prerequisites

Make sure dependencies are installed:

```bash
# From the project root
source venv/Scripts/activate  # Git Bash
# or
venv\Scripts\activate         # CMD/PowerShell

pip install -r requirements.txt
```

## Starting the UI

### Git Bash (recommended)
```bash
cd /c/Users/Bilal/Projects/turtlebot-path-traversal-optimization
source venv/Scripts/activate
streamlit run src/ui.py
```

### CMD / PowerShell
```cmd
cd C:\Users\Bilal\Projects\turtlebot-path-traversal-optimization
venv\Scripts\activate
streamlit run src/ui.py
```

### Alternative (no venv activation needed)
```bash
./venv/Scripts/python -m streamlit run src/ui.py
```

## Accessing the UI

After running the command, Streamlit will display:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open `http://localhost:8501` in your browser.

## Stopping the UI

Press `Ctrl+C` in the terminal where Streamlit is running.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | Run `streamlit run src/ui.py --server.port 8502` |
| Module not found | Activate venv and ensure `pip install -r requirements.txt` |
| Browser doesn't open | Manually navigate to `http://localhost:8501` |
| Encoding errors | Ensure files are UTF-8 (run `python -c "open('src/ui.py').read()"`) |
