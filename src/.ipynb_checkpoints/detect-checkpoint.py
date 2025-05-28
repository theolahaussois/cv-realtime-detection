import argparse  # Gestion des arguments en ligne de commande
import time     

import cv2       # OpenCV 
import mediapipe as mp 
from mediapipe.tasks.python import vision  # Module Vision de MediaPipe Tasks

BaseOptions           = mp.tasks.BaseOptions
ObjectDetector        = vision.ObjectDetector
ObjectDetectorOptions = vision.ObjectDetectorOptions
VisionRunningMode     = vision.RunningMode
MPImage               = mp.Image
ImageFormat           = mp.ImageFormat

def main(source=0, model_path="models/efficientdet_lite0.tflite"):
    """
    - source    : index de la caméra, on met 0 pour utiliser la caméra par défaut de l'ordinateur
    - model_path: chemin vers le modèle TFLite
    
    """

    # Configuration du détecteur d’objets
    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),  # Chargement du modèle
        running_mode=VisionRunningMode.VIDEO,                   # Réglage du mode vidéo sur flux continu
        max_results=10,                                         # Réglage du nombre maximum de détections : le nombre maximum d'objets détectés
        score_threshold=0.3,                                    # Seuil de confiance minimal, à partir de quel seuil on considère que c'est la classe de l'objets
        category_allowlist=[                                    # Liste des classes autorisées, afin de se concentrer sur quelques objets et avoir une meilleure performance.
            "person", "bottle", "cell phone", "toothbrush", "cup"
        ]
    )
    detector = ObjectDetector.create_from_options(options)  # Création du détecteur

    
    cap = cv2.VideoCapture(source) #Ouverture de la caméra

    prev_time = time.time() 
    while True:
        ret, frame = cap.read()  # Lecture d’une nouvelle image
        if not ret:  # Arrêt du programme
            break

        #Calcul du FPS en direct : on a mesuré le temps avant la lecture de l'image, puis on mesure après
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        #Prétraitement
        # Conversion du format BGR en RGB 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        timestamp = int(curr_time * 1000)  # Timestamp en millisecondes
        result = detector.detect_for_video(mp_image, timestamp)

        # Création des rectangles de détections, boucle sur chaque détection valide
        for det in result.detections:
            bbox = det.bounding_box
            # Calcul des coins du rectangle
            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            x2 = x1 + int(bbox.width)
            y2 = y1 + int(bbox.height)

            #Dessin du rectangle et du nom de la catégorie d'objet
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = det.categories[0].category_name
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 0)),             
                cv2.FONT_HERSHEY_SIMPLEX,           
                0.5,                                
                (0, 255, 0),                        
                2                                   
            )

        #Affichage du FPS à l’écran
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),  
            2
        )

        
        cv2.imshow("Détection objets", frame)
        # Arrêt si on appuie sur la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Libération
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    # Utilisation de argparse pour lancer detect.py simplement
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, default="0",
        help="Caméra (0) ou fichier vidéo"
    )
    parser.add_argument(
        "--model", type=str,
        default="models/efficientdet_lite0.tflite",
        help="Chemin vers le modèle TFLite"
    )
    args = parser.parse_args()

    # Conversion de source en int si c’est un chiffre
    src = int(args.source) if args.source.isdigit() else args.source
    main(source=src, model_path=args.model)
