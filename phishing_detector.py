"""Phishing Email Detection Model (Thiranex Task 3)
Scikit-learn: TF-IDF text features + URL/keyword features -> Logistic Regression.
Usage: python phishing_detector.py [path/to/emails.csv]
"""
import os, re, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
SHORTENERS = ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly")
KEYWORDS = ["urgent", "verify", "account", "password", "login", "suspended", "click",
            "bank", "confirm", "winner", "prize", "free", "limited time", "security alert"]

# Tiny demo data, used only if no CSV is found
DEMO = [
    ("URGENT: verify your account now http://192.168.4.7/login or it will be suspended", 1),
    ("Your bank password expired. Click http://bit.ly/3xYz to confirm your login", 1),
    ("Congratulations winner! Claim your free prize at http://free-prize.biz now", 1),
    ("Security alert: unusual sign-in. Confirm your account http://paypa1-secure.com", 1),
    ("Limited time! Click here to update your payment details http://tinyurl.com/abc", 1),
    ("Your mailbox is full. Login to verify http://mail-upgrade.xyz/@user", 1),
    ("Dear customer, your account is suspended. Verify at www.secure-bank-login.co", 1),
    ("You won a free gift card! Click http://bit.ly/gift to claim urgent", 1),
    ("Hi team, the meeting is moved to 3pm tomorrow. Agenda attached.", 0),
    ("Please find the lecture notes for Unit 2 attached. See you in class.", 0),
    ("Your order #4821 has shipped and will arrive on Friday.", 0),
    ("Reminder: project submission deadline is next Monday. Good luck!", 0),
    ("Thanks for your feedback, we'll review the report and get back to you.", 0),
    ("Lunch at 1pm? Let me know if the new cafe works for you.", 0),
    ("Minutes from yesterday's call are on our shared drive: https://docs.example.com/minutes", 0),
    ("Happy birthday! Hope you have a great day with family and friends.", 0),
]

def url_features(text):
    """Hand-made numeric features: URLs, suspicious links, keyword count."""
    urls = URL_RE.findall(text)
    low = text.lower()
    return [
        len(urls),
        int(any(re.match(r"https?://\d+\.\d+\.\d+\.\d+", u) for u in urls)),   # IP-based URL
        int(any(s in u.lower() for u in urls for s in SHORTENERS)),            # shortened URL
        int(any("@" in u for u in urls)),                                      # '@' inside URL
        int(any(u.lower().startswith("http://") for u in urls)),               # not HTTPS
        min(sum(k in low for k in KEYWORDS), 10),                              # phishing keywords
    ]

def load_data(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path).dropna()
        text_col = next(c for c in df.columns if any(k in c.lower() for k in ("text", "body", "content")))
        label_col = next(c for c in df.columns if any(k in c.lower() for k in ("type", "label", "class")))
        y = df[label_col].astype(str).str.lower().str.contains("phish|spam|1").astype(int)
        return df[text_col].astype(str), y
    print("[!] No dataset found - using tiny demo data. Pass a CSV path for real results.")
    df = pd.DataFrame(DEMO, columns=["text", "label"])
    return df["text"], df["label"]

def build_features(texts, vec, fit=False):
    tfidf = vec.fit_transform(texts) if fit else vec.transform(texts)
    return hstack([tfidf, csr_matrix([url_features(t) for t in texts])]).tocsr()

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "emails.csv"
    X, y = load_data(path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    model = LogisticRegression(max_iter=1000)
    model.fit(build_features(list(X_tr), vec, fit=True), y_tr)

    pred = model.predict(build_features(list(X_te), vec))
    print(f"\nAccuracy: {accuracy_score(y_te, pred) * 100:.2f}%")
    print("\nConfusion Matrix:\n", confusion_matrix(y_te, pred))
    print("\n", classification_report(y_te, pred, target_names=["Safe", "Phishing"], zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Safe", "Phishing"], cmap="Blues")
    plt.title("Phishing Email Detection - Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved confusion_matrix.png")

    # Try it on new emails
    for mail in ["Verify your password now: http://bit.ly/secure-login", "Notes for tomorrow's class are attached."]:
        label = model.predict(build_features([mail], vec))[0]
        print(f"[{'Phishing' if label else 'Safe'}] {mail}")

if __name__ == "__main__":
    main()
