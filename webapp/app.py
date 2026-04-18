# =============================================================================
# MODULE 4: FLASK WEB APPLICATION  (Enhanced)
# File: webapp/app.py
# Role: Browser-based interface for phishing email detection.
#       Now includes /predict_links and /predict_files endpoints so the
#       unified one-page UI can analyse email body, URLs, and attachments
#       in a single click.
# =============================================================================

import os
import re
import sys
import math
import pickle
from urllib.parse import urlparse
from flask import Flask, render_template, request, jsonify

# Ensure src/ is importable regardless of working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, '..', 'src')
sys.path.insert(0, SRC_DIR)

from preprocess import preprocess_single_email

app = Flask(__name__)

# ------------------------------------------------------------------
# Load model + vectorizer at startup
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'phishing_model.pkl')
VEC_PATH   = os.path.join(BASE_DIR, '..', 'models', 'tfidf_vectorizer.pkl')

model      = None
vectorizer = None


def load_artifacts():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VEC_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("[INFO] Model and vectorizer loaded successfully.")
    else:
        print("[WARN] Model files not found. Train the model first.")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


# ── EMAIL TEXT PREDICTION (original endpoint, unchanged) ──────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first by running '
                     '"python src/train_model.py dataset/emails.csv".'
        }), 503

    data       = request.get_json(force=True)
    email_text = data.get('email', '').strip()

    if not email_text:
        return jsonify({'error': 'No email text provided.'}), 400

    features   = preprocess_single_email(email_text, vectorizer)
    prediction = model.predict(features)[0]
    label      = 'Phishing' if prediction == 1 else 'Legitimate'

    confidence = None
    if hasattr(model, 'predict_proba'):
        proba      = model.predict_proba(features)[0]
        confidence = round(float(proba[prediction]) * 100, 2)
    elif hasattr(model, 'decision_function'):
        score      = model.decision_function(features)[0]
        confidence = round(100 / (1 + math.exp(-float(score))), 2)

    return jsonify({
        'label'     : label,
        'label_code': int(prediction),
        'confidence': confidence,
    })


# ── LINK / URL ANALYSIS ───────────────────────────────────────────
# Heuristic-based phishing URL detector.
# Checks for commonly abused patterns without requiring external APIs.

SUSPICIOUS_TLDS = {
    '.xyz', '.top', '.gq', '.ml', '.ga', '.cf', '.tk',
    '.click', '.loan', '.work', '.date', '.racing', '.win',
    '.download', '.stream', '.trade', '.review', '.accountant',
    '.science', '.party', '.faith', '.men', '.bid', '.webcam',
}

BRAND_KEYWORDS = [
    'paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
    'netflix', 'bank', 'secure', 'login', 'verify', 'account',
    'update', 'signin', 'password', 'confirm', 'ebay', 'irs',
    'chase', 'wellsfargo', 'citibank', 'hsbc',
]

IP_PATTERN = re.compile(r'https?://(\d{1,3}\.){3}\d{1,3}')


def analyse_url(url: str) -> dict:
    """Return a dict with url, suspicious (bool), and reason (str)."""
    reasons = []
    try:
        parsed = urlparse(url if url.startswith('http') else 'http://' + url)
        host   = parsed.hostname or ''
        path   = parsed.path.lower()
        full   = url.lower()

        # 1. IP address instead of domain
        if IP_PATTERN.match(url):
            reasons.append("Uses raw IP address instead of a domain name")

        # 2. Suspicious TLD
        for tld in SUSPICIOUS_TLDS:
            if host.endswith(tld):
                reasons.append(f"Uses suspicious TLD '{tld}'")
                break

        # 3. Too many subdomains (e.g. secure.login.paypal.phish.com)
        parts = host.split('.')
        if len(parts) > 4:
            reasons.append(f"Excessive subdomain nesting ({len(parts)} levels)")

        # 4. Brand keyword in subdomain but not the registered domain
        registered = '.'.join(parts[-2:]) if len(parts) >= 2 else host
        subdomain  = '.'.join(parts[:-2]) if len(parts) > 2 else ''
        for brand in BRAND_KEYWORDS:
            if brand in subdomain and brand not in registered:
                reasons.append(f"Brand keyword '{brand}' appears in subdomain, not domain")
                break

        # 5. Very long URL
        if len(url) > 200:
            reasons.append(f"Unusually long URL ({len(url)} chars)")

        # 6. @ symbol in URL (obscures real destination)
        if '@' in url:
            reasons.append("Contains '@' symbol, which can hide the real destination")

        # 7. Multiple redirectors / obfuscators
        if full.count('http') > 1:
            reasons.append("URL contains nested HTTP, possible redirect chain")

        # 8. Encoded characters (hex or percent)
        if '%' in url or '0x' in full:
            reasons.append("URL contains encoded/obfuscated characters")

        # 9. Deceptive keywords in path
        for kw in ['login', 'signin', 'verify', 'secure', 'update', 'confirm', 'bank', 'password']:
            if kw in path:
                reasons.append(f"Suspicious keyword '{kw}' in URL path")
                break

        suspicious = bool(reasons)
        return {
            'url'       : url,
            'suspicious': suspicious,
            'reason'    : '; '.join(reasons) if reasons else 'No obvious phishing indicators detected',
        }

    except Exception as e:
        return {'url': url, 'suspicious': False, 'reason': f'Could not parse URL: {e}'}


@app.route('/predict_links', methods=['POST'])
def predict_links():
    data  = request.get_json(force=True)
    links = data.get('links', [])
    if not isinstance(links, list):
        return jsonify({'error': 'links must be a list'}), 400
    results = [analyse_url(str(u)) for u in links if str(u).strip()]
    return jsonify({'results': results})


# ── FILE / ATTACHMENT ANALYSIS ────────────────────────────────────
# Heuristic checks on filename, extension, and size.
# Deep content scanning requires antivirus integration (future work).

DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.ps1', '.vbs', '.vbe', '.js', '.jse',
    '.wsf', '.wsh', '.msi', '.msp', '.scr', '.hta', '.pif', '.cpl',
    '.dll', '.sys', '.reg', '.inf',
}

DOUBLE_EXTENSION_PATTERN = re.compile(
    r'\.(pdf|doc|docx|xls|xlsx|txt|jpg|png)\.(exe|bat|cmd|scr|vbs|js)$',
    re.IGNORECASE,
)

SUSPICIOUS_NAME_KEYWORDS = [
    'invoice', 'payment', 'urgent', 'verify', 'confirm', 'bank',
    'refund', 'update', 'password', 'credentials', 'account',
]


def analyse_file(filename: str, size_bytes: int) -> dict:
    """Return a dict with filename, suspicious (bool), and reason (str)."""
    reasons  = []
    name_low = filename.lower()
    ext      = os.path.splitext(name_low)[1]

    # 1. Directly executable extension
    if ext in DANGEROUS_EXTENSIONS:
        reasons.append(f"Dangerous file type '{ext}' — may be executable malware")

    # 2. Double extension trick (resume.pdf.exe)
    if DOUBLE_EXTENSION_PATTERN.search(name_low):
        reasons.append("Double extension detected — file extension is being disguised")

    # 3. Suspicious name keywords
    for kw in SUSPICIOUS_NAME_KEYWORDS:
        if kw in name_low:
            reasons.append(f"Filename contains suspicious keyword '{kw}'")
            break

    # 4. Password-protected or encrypted archive hint
    if ext == '.zip' and any(kw in name_low for kw in ['pass', 'protected', 'encrypted']):
        reasons.append("ZIP filename hints at password protection — common in malware delivery")

    # 5. Unusually large file for document
    if ext in {'.doc', '.docx', '.xls', '.xlsx', '.pdf', '.txt'} and size_bytes > 5 * 1024 * 1024:
        reasons.append(f"Document is unusually large ({size_bytes // (1024*1024)} MB) — may contain embedded payload")

    suspicious = bool(reasons)
    return {
        'filename'  : filename,
        'suspicious': suspicious,
        'reason'    : '; '.join(reasons) if reasons else 'No obvious malware indicators detected',
    }


@app.route('/predict_files', methods=['POST'])
def predict_files():
    files   = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
    results = [analyse_file(f.filename, len(f.read())) for f in files]
    return jsonify({'results': results})


# ------------------------------------------------------------------
# Sample emails endpoint
# ------------------------------------------------------------------
SAMPLES = {
    'phishing': (
        "URGENT: Your account has been suspended! Click here immediately to verify "
        "your identity and restore access. Failure to act within 24 hours will result "
        "in permanent deletion. Login now: http://secure-verify-account.xyz/login"
    ),
    'legitimate': (
        "Hi Sarah, just a reminder about tomorrow's 10 AM project sync. "
        "We'll be reviewing the Q3 roadmap and aligning on sprint priorities. "
        "Please review the attached slides before the meeting. See you then! – Mark"
    ),
}

@app.route('/sample/<sample_type>')
def sample_email(sample_type: str):
    text = SAMPLES.get(sample_type)
    if text is None:
        return jsonify({'error': 'Unknown sample type.'}), 404
    return jsonify({'email': text})


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
if __name__ == '__main__':
    load_artifacts()
    app.run(debug=True, host='0.0.0.0', port=5000)
