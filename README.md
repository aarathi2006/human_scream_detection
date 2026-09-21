# Human Scream Detection 🎙️🚨

The **Human Scream Detection System** is an AI-powered safety application designed to detect human screams in real time and classify them as **Positive** (normal / excitement) or **Negative** (distress / emergency). The system focuses on improving personal safety by automatically triggering emergency alerts when a dangerous scream is detected.

This project uses machine learning and audio signal processing to analyze real-time audio input and determine the emotional intensity and urgency of the scream.

---

## 📌 Key Features

- 🎤 Real-time audio recording and processing
- 🧠 Binary classification of screams:
  - ✅ **Positive Scream** — non-dangerous / excitement / fun
  - ❌ **Negative Scream** — danger / pain / distress
- 📩 Automatic SMS alerts sent to pre-registered caretakers
- 📍 GPS location tracking of the person who screamed
- 🟢 Live system status display (Recording Start / Stop indicators)
- 🔐 Secure user registration and caretaker contact setup

![System Preview](https://github.com/user-attachments/assets/ddf226da-3be8-494d-8961-7a2e65aa3ee0)

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 🐍 |
| Machine Learning | CNN / LSTM-based audio classification |
| Frameworks | TensorFlow / Keras |
| Audio Processing | Librosa (MFCC feature extraction) |
| Web Framework | Flask |
| Alerts | Twilio API (SMS) |
| Location | GPS / Geolocation API |

---

## ⚙️ How the System Works

1. The system captures **live audio** from the microphone.
2. The audio is converted into **feature vectors** using signal processing techniques such as **MFCC**.
3. A trained **ML model** predicts whether the scream is **positive** or **negative**.
4. If a **negative scream** is detected:
   - An alert SMS is sent to the caretaker.
   - The current **GPS coordinates** of the user are attached to the message.
5. If the scream is **positive**, no alert is sent.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/aarathi2006/human_scream_detection.git
cd human_scream_detection
python3 app.py

