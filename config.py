import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
    UPLOAD_FOLDER = 'uploads'
    DATABASE = os.path.join('instance', 'app.db')
    MODEL_PATH = os.path.join('models', 'isolation_forest.pkl')
    ALLOWED_EXTENSIONS = {'csv'}