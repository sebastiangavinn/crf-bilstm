"""
Script untuk menjalankan API server
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload saat ada perubahan (untuk development)
        log_level="info"
    )

