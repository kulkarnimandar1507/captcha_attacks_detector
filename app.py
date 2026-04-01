import os
import re
import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, g, abort
)
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from sklearn.ensemble import IsolationForest

from pathlib import Path

# Absolute path to the project root
BASE_DIR = Path(__file__).resolve().parent

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
class Config:
    # SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    # MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
    # UPLOAD_FOLDER = 'uploads'
    # DATABASE = os.path.join('instance', 'app.db')
    # MODEL_PATH = os.path.join('models', 'isolation_forest.pkl')
    # ALLOWED_EXTENSIONS = {'csv'}
    class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    DATABASE = BASE_DIR / 'instance' / 'app.db'
    MODEL_PATH = BASE_DIR / 'models' / 'isolation_forest.pkl'
    ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config.from_object(Config)

# -------------------------------------------------------------------
# Flask-Login setup
# -------------------------------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# -------------------------------------------------------------------
# Database helpers
# -------------------------------------------------------------------
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            total_rows INTEGER NOT NULL,
            suspicious_count INTEGER NOT NULL,
            suspicious_percent REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attempt_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            solve_time REAL NOT NULL,
            retry_count INTEGER NOT NULL,
            attempts_in_window INTEGER NOT NULL,
            consistency_score REAL NOT NULL,
            normalized_solve_time REAL NOT NULL,
            anomaly_score REAL NOT NULL,
            label TEXT NOT NULL,
            FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
        )
    ''')
    db.commit()

# Initialize DB at startup (if tables missing)
with app.app_context():
    init_db()

# -------------------------------------------------------------------
# User model for Flask-Login
# -------------------------------------------------------------------
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    @staticmethod
    def get(user_id):
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if not user:
            return None
        return User(user['id'], user['username'], user['password_hash'])

    @staticmethod
    def find_by_username(username):
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if not user:
            return None
        return User(user['id'], user['username'], user['password_hash'])

    @staticmethod
    def create(username, password, email=None):
        db = get_db()
        password_hash = generate_password_hash(password)
        try:
            db.execute(
                'INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
                (username, password_hash, email)
            )
            db.commit()
            return True
        except sqlite3.IntegrityError:
            return False

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# -------------------------------------------------------------------
# Password validation function
# -------------------------------------------------------------------
def validate_password_strength(password):
    """Check password complexity."""
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r'[A-Z]', password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r'[a-z]', password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r'\d', password):
        return "Password must contain at least one digit."
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return "Password must contain at least one special character (!@#$%^&* etc.)."
    return None

# -------------------------------------------------------------------
# ML Model Manager (unchanged)
# -------------------------------------------------------------------
class ModelManager:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            model_path = app.config['MODEL_PATH']
            if os.path.exists(model_path):
                cls._model = joblib.load(model_path)
            else:
                cls._model = cls._train_model()
                joblib.dump(cls._model, model_path)
        return cls._model

    @staticmethod
    def _train_model():
        np.random.seed(42)
        n_human = 1000
        n_bot = 1000
        human_data = {
            'solve_time': np.random.uniform(2, 10, n_human),
            'retry_count': np.random.poisson(1, n_human).clip(0, 5),
            'attempts_in_window': np.random.poisson(3, n_human).clip(1, 10),
            'consistency_score': np.random.uniform(0.7, 1.0, n_human)
        }
        bot_data = {
            'solve_time': np.concatenate([
                np.random.uniform(0.1, 1.0, n_bot//2),
                np.random.uniform(1, 5, n_bot//2)
            ]),
            'retry_count': np.concatenate([
                np.random.poisson(10, n_bot//2).clip(5, 30),
                np.random.poisson(2, n_bot//2).clip(0, 5)
            ]),
            'attempts_in_window': np.concatenate([
                np.random.poisson(20, n_bot//2).clip(10, 100),
                np.random.poisson(5, n_bot//2).clip(1, 20)
            ]),
            'consistency_score': np.concatenate([
                np.random.uniform(0.0, 0.3, n_bot//2),
                np.random.uniform(0.9, 1.0, n_bot//2)
            ])
        }
        df_human = pd.DataFrame(human_data)
        df_bot = pd.DataFrame(bot_data)
        df = pd.concat([df_human, df_bot], ignore_index=True)
        df['normalized_solve_time'] = df['solve_time'] / (df['attempts_in_window'] + 1)
        features = ['solve_time', 'retry_count', 'attempts_in_window',
                    'consistency_score', 'normalized_solve_time']
        X = df[features]
        model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100, bootstrap=True)
        model.fit(X)
        return model

    @staticmethod
    def predict(df_features):
        model = ModelManager.get_model()
        feature_cols = ['solve_time', 'retry_count', 'attempts_in_window',
                        'consistency_score', 'normalized_solve_time']
        X = df_features[feature_cols].values
        preds = model.predict(X)
        scores = model.decision_function(X)
        labels = ['Human' if p == 1 else 'Suspicious' for p in preds]
        return labels, scores.tolist()

# -------------------------------------------------------------------
# File upload helpers (unchanged)
# -------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return None, f"Could not read CSV file: {str(e)}"
    required_columns = ['solve_time', 'retry_count', 'attempts_in_window', 'consistency_score']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}"
    if df.empty:
        return None, "CSV file is empty."
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[required_columns].isnull().any().any():
        return None, "Non‑numeric or missing values found."
    df['solve_time'] = df['solve_time'].clip(lower=0)
    df['retry_count'] = df['retry_count'].clip(lower=0)
    df['attempts_in_window'] = df['attempts_in_window'].clip(lower=0)
    df['consistency_score'] = df['consistency_score'].clip(lower=0, upper=1)
    df['normalized_solve_time'] = df['solve_time'] / (df['attempts_in_window'] + 1)
    if df['normalized_solve_time'].isnull().any() or np.isinf(df['normalized_solve_time']).any():
        return None, "Error computing normalized_solve_time."
    return df, None

def analyze_df(df):
    labels, scores = ModelManager.predict(df)
    df['label'] = labels
    df['anomaly_score'] = scores
    total = len(df)
    suspicious_count = (df['label'] == 'Suspicious').sum()
    suspicious_percent = round((suspicious_count / total) * 100, 2) if total > 0 else 0
    summary = {
        'total_rows': total,
        'suspicious_count': suspicious_count,
        'suspicious_percent': suspicious_percent
    }
    return df, summary

def save_upload_results(user_id, original_filename, stored_filename, df, summary):
    db = get_db()
    cursor = db.cursor()
    upload_time = datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO uploads (user_id, original_filename, stored_filename, upload_time,
                             total_rows, suspicious_count, suspicious_percent)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, original_filename, stored_filename, upload_time,
          summary['total_rows'], summary['suspicious_count'], summary['suspicious_percent']))
    upload_id = cursor.lastrowid
    for _, row in df.iterrows():
        cursor.execute('''
            INSERT INTO attempt_results
            (upload_id, solve_time, retry_count, attempts_in_window,
             consistency_score, normalized_solve_time, anomaly_score, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            upload_id,
            row['solve_time'],
            row['retry_count'],
            row['attempts_in_window'],
            row['consistency_score'],
            row['normalized_solve_time'],
            row['anomaly_score'],
            row['label']
        ))
    db.commit()
    return upload_id

def fetch_uploads(user_id, limit=50):
    db = get_db()
    return db.execute('''
        SELECT id, original_filename, upload_time, total_rows,
               suspicious_count, suspicious_percent
        FROM uploads
        WHERE user_id = ?
        ORDER BY upload_time DESC
        LIMIT ?
    ''', (user_id, limit)).fetchall()

def fetch_upload_results(upload_id, user_id):
    db = get_db()
    upload = db.execute('SELECT * FROM uploads WHERE id = ? AND user_id = ?',
                        (upload_id, user_id)).fetchone()
    if not upload:
        return None, None
    rows = db.execute('''
        SELECT solve_time, retry_count, attempts_in_window,
               consistency_score, normalized_solve_time, anomaly_score, label
        FROM attempt_results
        WHERE upload_id = ?
    ''', (upload_id,)).fetchall()
    return upload, rows

# -------------------------------------------------------------------
# Authentication Routes
# -------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.find_by_username(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        email = request.form.get('email')

        if not username or not password:
            flash('Username and password required.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        else:
            # Validate password strength
            error_msg = validate_password_strength(password)
            if error_msg:
                flash(error_msg, 'error')
            else:
                if User.create(username, password, email):
                    flash('Account created! Please log in.', 'success')
                    return redirect(url_for('login'))
                else:
                    flash('Username already taken.', 'error')
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# -------------------------------------------------------------------
# Protected Routes (require login)
# -------------------------------------------------------------------
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/analyze_csv', methods=['POST'])
@login_required
def analyze_csv():
    if 'csv_file' not in request.files:
        flash('No file selected.', 'error')
        return redirect(url_for('index'))
    file = request.files['csv_file']
    if file.filename == '':
        flash('Empty filename.', 'error')
        return redirect(url_for('index'))
    if not allowed_file(file.filename):
        flash('Only .csv files are allowed.', 'error')
        return redirect(url_for('index'))

    original_filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    stored_filename = f"{timestamp}_{original_filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
    file.save(save_path)

    df, error = validate_csv(save_path)
    if error:
        os.remove(save_path)
        flash(error, 'error')
        return redirect(url_for('index'))

    df, summary = analyze_df(df)
    upload_id = save_upload_results(current_user.id, original_filename, stored_filename, df, summary)

    return redirect(url_for('result', upload_id=upload_id))

@app.route('/analyze_manual', methods=['POST'])
@login_required
def analyze_manual():
    try:
        solve_time = float(request.form.get('solve_time', 0))
        retry_count = int(request.form.get('retry_count', 0))
        attempts_in_window = int(request.form.get('attempts_in_window', 0))
        consistency_score = float(request.form.get('consistency_score', 0))
    except ValueError:
        flash('Invalid numeric input in manual form.', 'error')
        return redirect(url_for('index'))

    solve_time = max(0, solve_time)
    retry_count = max(0, retry_count)
    attempts_in_window = max(0, attempts_in_window)
    consistency_score = max(0, min(1, consistency_score))
    normalized_solve_time = solve_time / (attempts_in_window + 1)

    df = pd.DataFrame([{
        'solve_time': solve_time,
        'retry_count': retry_count,
        'attempts_in_window': attempts_in_window,
        'consistency_score': consistency_score,
        'normalized_solve_time': normalized_solve_time
    }])

    labels, scores = ModelManager.predict(df)
    label = labels[0]
    score = scores[0]

    return render_template('index.html',
                           manual_result={
                               'label': label,
                               'anomaly_score': round(score, 4),
                               'solve_time': solve_time,
                               'retry_count': retry_count,
                               'attempts_in_window': attempts_in_window,
                               'consistency_score': consistency_score
                           })

@app.route('/result/<int:upload_id>')
@login_required
def result(upload_id):
    upload, rows = fetch_upload_results(upload_id, current_user.id)
    if upload is None:
        abort(404, description="Upload not found.")
    # Convert suspicious_count if bytes (just in case)
    if isinstance(upload['suspicious_count'], bytes):
        upload = dict(upload)
        try:
            upload['suspicious_count'] = int.from_bytes(upload['suspicious_count'], 'little')
        except:
            upload['suspicious_count'] = 0
    return render_template('result.html', upload=upload, rows=rows)

@app.route('/history')
@login_required
def history():
    uploads = fetch_uploads(current_user.id)
    # Convert any bytes in suspicious_count (for safety)
    converted = []
    for up in uploads:
        up_dict = dict(up)
        if isinstance(up_dict['suspicious_count'], bytes):
            try:
                up_dict['suspicious_count'] = int.from_bytes(up_dict['suspicious_count'], 'little')
            except:
                up_dict['suspicious_count'] = 0
        converted.append(up_dict)
    return render_template('history.html', uploads=converted)

@app.route('/history/<int:upload_id>')
@login_required
def view_upload(upload_id):
    upload, rows = fetch_upload_results(upload_id, current_user.id)
    if upload is None:
        abort(404, description="Upload not found.")
    if isinstance(upload['suspicious_count'], bytes):
        upload = dict(upload)
        try:
            upload['suspicious_count'] = int.from_bytes(upload['suspicious_count'], 'little')
        except:
            upload['suspicious_count'] = 0
    return render_template('view_upload.html', upload=upload, rows=rows)

@app.route('/download/<int:upload_id>')
@login_required
def download(upload_id):
    db = get_db()
    upload = db.execute('SELECT stored_filename, original_filename FROM uploads WHERE id = ? AND user_id = ?',
                        (upload_id, current_user.id)).fetchone()
    if not upload:
        abort(404, description="Upload not found.")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], upload['stored_filename'])
    if not os.path.exists(file_path):
        abort(404, description="File not found.")
    return send_file(file_path, as_attachment=True, download_name=upload['original_filename'])

# -------------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------------
@app.errorhandler(400)
def bad_request(e):
    return render_template('error.html', message="Bad request. Please check your input."), 400

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="The requested page was not found."), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Internal error: {e}")
    return render_template('error.html', message="An internal error occurred. Please try again later."), 500

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
