# 📩 SMS Spam Classifier

This is a **machine learning-powered SMS Spam Classifier** that detects whether a message is **Spam or Not Spam** using **Natural Language Processing (NLP)**.

---

## 📂 Project Structure
```plaintext
sms-spam-classifier/
│
└── images/           # Folder for result images
    ├── Home.png
    ├── Not_Spam.png
    ├── Span.png
├── app.py             # Main Streamlit application
├── model.pkl         # Trained Naive Bayes model
├── vectorizer.pkl    # TF-IDF vectorizer
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation

```

---

## 💪 Screenshots
### Home Page
![Home](Images/Home.png)

### Not Spam Example
![Not Spam](Images/Not_Spam.png)

### Spam Example
![Spam](Images/Spam.png)

---

## ⚙️ Setup Instructions

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
```

### **2️⃣ Create a Virtual Environment & Activate It**
#### Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
#### Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the Streamlit App**
```bash
streamlit run app.py
```
The app should open in your web browser at **http://localhost:8501**.

---

## 🛠️ Train a New Model (Optional)
If you want to train the model from scratch, modify `train.py` to include training logic and run:
```bash
python train.py
```
Then, save the trained model as `model.pkl`.

---

## 🚀 Deploying on Render

### **1️⃣ Ensure Dependencies are Listed in `requirements.txt`**
```txt
streamlit
pandas
scikit-learn
nltk
pickle5
```

### **2️⃣ Push the Project to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/sms-spam-classifier.git
git push -u origin main
```

### **3️⃣ Deploy on Render**
- Go to [Render](https://render.com/).
- Create a new Web Service.
- Connect your GitHub repository.
- Set the Start Command to:
```bash
streamlit run app.py
```
- Deploy and get the live URL.

---

## 📧 Usage
- Open the web app.
- Enter an SMS or email message.
- Click **Predict** to check if it’s spam.

---

## 👉 Example Spam Messages
```plaintext
"Congratulations! You won a free iPhone. Click here to claim now!"
"URGENT! Your bank account has been locked. Verify now!"
"You have been selected for a $500 Amazon gift card. Reply YES to claim."
```

---

## 🐝 License
This project is open-source and available under the **MIT License**.

---




