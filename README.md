# Phishing-email-detector
For my Thiranex Project.



# Phishing Email Detection Model 

Classifies emails as **Phishing** or **Safe** using Scikit-learn.

## How it works
- TF-IDF on the email text
- Extra features: URL count, IP-based URLs, shortened links, `@` in URL, non-HTTPS links, phishing keyword count
- Logistic Regression classifier
- Prints accuracy + confusion matrix, saves `confusion_matrix.png`

## Run
```bash
pip install -r requirements.txt
python phishing_detector.py emails.csv
```
Dataset: any CSV with an email-text column and a label column (e.g. Kaggle "Phishing Email Dataset").
