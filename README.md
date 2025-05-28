# CV Real-Time Detection

Ce projet implémente un pipeline de détection d’objets en temps réel à partir d’un flux vidéo, utilisant :

- **MediaPipe ObjectDetector**   
- **OpenCV** 
- **Python 3.8+**

---

## 📂 Structure du dépôt

```text
cv-realtime-detection/
├── models/
│   └── efficientdet_lite0_int8.tflite  
├── src/
│   └── detect.py                       
├── rapport_Lahaussois.pdf                                     
├── requirements.txt               
└── README.md
```

## 💻 Commandes

```
git clone https://github.com/theolahaussois/cv-realtime-detection.git

cd cv-realtime-detection

pip install -r requirements.txt

python src/detect.py
