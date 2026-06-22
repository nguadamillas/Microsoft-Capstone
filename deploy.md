Deployment notes

Streamlit Cloud
---------------
- Create a new app on Streamlit Cloud and connect your GitHub repo.
- Set the "Main file" to `app/streamlit_designed.py` and the Python version to 3.11.
- Add a secret `ANTHROPIC_API_KEY` in the app settings.

Docker (generic)
----------------
Build and run locally:

```bash
docker build -t ted-procurement:latest .
docker run -p 8501:8501 ted-procurement:latest
```

Azure Web App for Containers
----------------------------
- Push Docker image to ACR or Docker Hub.
- Create Web App and configure container settings to use the image.
- Set env var `ANTHROPIC_API_KEY` in the App Service settings.
