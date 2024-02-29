import numpy as np
import cv2
import os
import tkinter
from tkinter import *
import pyautogui
from PIL import Image
import argparse
import pyodbc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=False, default="yolo_model",
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Read YOLO model and configuration
weightsPath = os.path.sep.join([args["yolo"], "custom.weights"])
configPath = os.path.sep.join([args["yolo"], "custom.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Database connection
server = 'LAPTOP-NO435VT2\MSSQLSERVER01'
database = 'DefaultDatabase'
cnxn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database}')
cursor = cnxn.cursor()

# Sample dataset for KNN training (replace this with your actual dataset)
data = np.array([
    [0.2, 2, "hole"],
    [3.0, 2.0, "leisure"],
    [1.5, 1.0, "hole"],
    [2.5, 3.5, "leisure"],
    [1.0, 3.0, "leisure"],  # Adjusted for "leisure" class with height > 1.5
    # Add more examples with corresponding height, width, and labels
])

# Split data into features (height and width) and labels
X = data[:, :2].astype(float)
y = data[:, 2]

# Encode labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))
knn_classifier.fit(X_train, y_train)

def classify_object_knn(width_cm, height_cm):
    features = np.array([[width_cm, height_cm]])
    prediction = knn_classifier.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    # Check if both width and height are between 0 and 0.5
    if 0 <= width_cm <= 0.2 and 0 <= height_cm <= 0.2:
        return "hole", 0  # Assigning 0 points for objects classified as "hole"
    else:
        max_dimension = max(width_cm, height_cm)

        if max_dimension <= 7.5:
            points_awarded = 1
        elif max_dimension <= 15:
            points_awarded = 2
        elif max_dimension <= 23:
            points_awarded = 3
        elif max_dimension <= 100:
            points_awarded = 4
        else:
            points_awarded = 0

        return predicted_label, points_awarded

def insert(width_cm, height_cm, anomaly_image_data, full_screen_image_data, object_name):
    width_cm = float(width_cm)
    height_cm = float(height_cm)

    try:
        current_date = datetime.now()
        classification_result, points_awarded = classify_object_knn(width_cm, height_cm)

        sql = "INSERT INTO anomalie (ObjectName, height, width, Date, Classification, laise, PointsAwarded) VALUES (?, ?, ?, ?, ?, ?, ?)"
        val = (object_name, height_cm, width_cm, current_date, classification_result, None, points_awarded)

        with pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database}') as cnxn:
            with cnxn.cursor() as cursor:
                cursor.execute(sql, val)
                cnxn.commit()

                row_count = cursor.rowcount
                if row_count == 0:
                    print("Error: Values not inserted into anomalie")
                else:
                    print("Values inserted successfully into anomalie")

    except pyodbc.Error as e:
        print(f"Error: {e}")
        print(f"Anomaly image size: {len(anomaly_image_data)} bytes")
        print(f"Full screen image size: {len(full_screen_image_data)} bytes")

def browse():
    path = tkinter.filedialog.askopenfilename()
    if len(path) > 0:
        print(path)
        image = cv2.imread(path)
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = (0, 255, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                zone_size_y = 10 
                zone_size_x = 10
                height_cm = h / image.shape[0] * zone_size_y
                width_cm = w / image.shape[1] * zone_size_x
                height_cm = h / image.shape[0] * zone_size_y

                classification_result, _ = classify_object_knn(width_cm, height_cm)

                text = f"W: {width_cm:.2f} cm, H: {height_cm:.2f}, Classification: {classification_result}"
                cv2.putText(image, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                print(text)

                save_folder = r"C:\PFE\Fabric-Stain-Detection-main\captures\images"
                anomaly_screenshot = pyautogui.screenshot(region=(int(x), int(y), int(w), int(h)))
                image_path = os.path.join(save_folder, f"anomaly_{len(os.listdir(save_folder))}.png")
                print(f"Saving anomaly screenshot to: {image_path}")
                anomaly_screenshot.save(image_path, quality=95)

                with open(image_path, "rb") as file:
                    image_data = file.read()

                full_screen_screenshot = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                full_screen_image_path = os.path.join(save_folder, f"full_screen_{len(os.listdir(save_folder))}.png")
                full_screen_screenshot.save(full_screen_image_path, quality=95)

                with open(full_screen_image_path, "rb") as file:
                    full_screen_image_data = file.read()

                insert(width_cm, height_cm, image_data, full_screen_image_data, f"Object_{len(os.listdir(save_folder))}")

        image = cv2.resize(image, (640, 480))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.moveWindow("Image", 0, 0)


            

root = Tk()
root.geometry("600x300")

label = Label(root, text="WIC MIC fabric default detection using IA", font=("Courier", 14, "italic", "bold"))
label.place(x=310, y=40, anchor="center")

confidence_values = [0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
confidence_var = StringVar(root)
confidence_var.set(confidence_values[5])

def update_confidence(value):
    args["confidence"] = value

dropdown_label = Label(root, text="Choose confidence")
dropdown_label.place(x=100, y=150, anchor="center")
dropdown = OptionMenu(root, confidence_var, *confidence_values, command=update_confidence)
dropdown.place(x=200, y=150, anchor="center")

btn = Button(root, text="Browse", command=browse)
btn.place(x=300, y=150, anchor="center")
def capture_and_detect():
    cap = cv2.VideoCapture(0)
    anomaly_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    x = max(0, x)
                    y = max(0, y)
                    width = min(frame.shape[1] - x, width)
                    height = min(frame.shape[0] - y, height)

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
                    zone_size_x = 10
                    zone_size_y = 10
                    width_cm = width / frame.shape[1] * zone_size_x
                    height_cm = height / frame.shape[0] * zone_size_y

                    classification_result, points_awarded = classify_object_knn(width_cm, height_cm)

                    text = f"W: {width_cm:.2f} cm, H: {height_cm:.2f}, Classification: {classification_result}, Points: {points_awarded}"
                    cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    save_folder = r"C:\PFE\Fabric-Stain-Detection-main\captures"

                    full_screen_screenshot = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    full_screen_image_path = os.path.join(save_folder, f"full_screen_{anomaly_count}.png")
                    full_screen_screenshot.save(full_screen_image_path, quality=95)

                    with open(full_screen_image_path, "rb") as file:
                        full_screen_image_data = file.read()

                    anomaly_region = frame[y:y + height, x:x + width]
                    anomaly_screenshot = Image.fromarray(cv2.cvtColor(anomaly_region, cv2.COLOR_BGR2RGB))
                    anomaly_screenshot_path = os.path.join(save_folder, f"anomaly_{anomaly_count}.png")
                    anomaly_screenshot.save(anomaly_screenshot_path, quality=95)

                    with open(anomaly_screenshot_path, "rb") as file:
                        anomaly_screenshot_data = file.read()

                    insert(width_cm, height_cm, anomaly_screenshot_data, full_screen_image_data, f"Object_{anomaly_count}")

                    anomaly_count += 1
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

btn_camera = Button(root, text="Camera", command=capture_and_detect)
btn_camera.place(x=400, y=150, anchor="center")

root.mainloop()
