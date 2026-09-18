from fastapi import FastAPI

app = FastAPI(
    title="Enterprise AI Knowledge & Support Agent",
    description="AI-powered enterprise knowledge and support platform",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Enterprise AI Agent is running!",
        "status": "healthy",
    }