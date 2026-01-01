# main.py
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import sys

# Ignore minor warnings to keep the console clean
warnings.filterwarnings("ignore")

# ==========================================
# --- STEP 1: CONFIGURATION & GLOBALS ---
# ==========================================
DATASET_DIR = 'dataset1'
TRAIN_DIR = os.path.join(DATASET_DIR, 'Train')
VAL_DIR = os.path.join(DATASET_DIR, 'Validation')
TEST_DIR = os.path.join(DATASET_DIR, 'Test')

CATEGORIES = ['WithMask', 'WithoutMask']
IMG_SIZE = 64
PROB_THRESHOLD = 0.75
MODEL_FILENAME = 'mask_detector_all_models.pkl'

# Global variables to hold state across functions
models_pool = {}
scaler = None
pca = None
active_model_name = ""
current_model = None
X_test_pca = None  # Transformed test data for confusion matrix
y_test = None  # Test labels


# ========================================================
# --- STEP 2: HELPER FUNCTIONS (PRE-PROCESSING) ---
# ========================================================

def extract_features(img_path):
    """
    Reads an image, applies CLAHE for lighting correction, resizes it,
    and extracts Histogram of Oriented Gradients (HOG) features.
    """
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None or img_array.size == 0: return None

    # CLAHE for lighting correction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img_array)
    resized_array = cv2.resize(enhanced_img, (IMG_SIZE, IMG_SIZE))

    # HOG Extraction
    features = hog(resized_array, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features


def load_images_from_directory(directory):
    """
    Iterates through a specific directory, loads images,
    and compiles their feature vectors and labels.
    """
    data, labels = [], []
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found!")
        return np.array([]), np.array([])

    for category in CATEGORIES:
        path = os.path.join(directory, category)
        class_num = CATEGORIES.index(category)
        if not os.path.exists(path): continue

        print(f"Processing {category} in {os.path.basename(directory)}...")
        for img_name in os.listdir(path):
            try:
                features = extract_features(os.path.join(path, img_name))
                if features is not None:
                    data.append(features)
                    labels.append(class_num)
            except:
                continue
    return np.array(data), np.array(labels)


# ========================================================
# --- STEP 3: TRAINING & SAVING LOGIC ---
# ========================================================

def train_system():
    """
    Loads data, trains models, and saves everything to disk.
    This is only called if no model exists or user selects 'Retrain'.
    """
    global models_pool, scaler, pca, active_model_name, current_model, X_test_pca, y_test

    print("\n" + "=" * 50)
    print("STARTING FULL TRAINING PIPELINE")
    print("=" * 50)

    # 1. Load Data
    print("Loading Training Data...")
    X_train_raw, y_train_local = load_images_from_directory(TRAIN_DIR)
    print("Loading Validation Data...")
    X_val_raw, y_val_local = load_images_from_directory(VAL_DIR)
    print("Loading Test Data...")
    X_test_raw, y_test_local = load_images_from_directory(TEST_DIR)

    if len(X_train_raw) == 0:
        print("Error: No training data found. Aborting.")
        return False

    # 2. Scale & PCA
    print("\nApplying StandardScaler and PCA...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    if len(X_val_raw) > 0:
        X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    # X_val_pca = pca.transform(X_val_scaled) if len(X_val_raw) > 0 else []
    # (Validation set logic kept simple here)
    X_test_pca = pca.transform(X_test_scaled)

    # Update global label reference for testing
    y_test = y_test_local

    # 3. Train Models
    models_pool = {
        'SVM': SVC(kernel='rbf', C=0.8, probability=True, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, C=0.01)
    }

    results = {}
    active_model_name = ""
    best_acc = -1

    print("\nTraining Models...")
    for name, clf in models_pool.items():
        print(f"Training {name}...")
        clf.fit(X_train_pca, y_train_local)

        preds = clf.predict(X_test_pca)
        acc = accuracy_score(y_test_local, preds)
        results[name] = acc * 100
        print(f"-> {name} Test Accuracy: {acc * 100:.2f}%")

        # Save individual confusion matrix images
        cm = confusion_matrix(y_test_local, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(f'confusion_matrix_{name}.png')
        plt.close()

        if acc > best_acc:
            best_acc = acc
            active_model_name = name

    current_model = models_pool[active_model_name]
    print(f"\n🏆 Best Model: {active_model_name}")

    # 4. Save Everything
    save_data = {
        'models': models_pool,
        'scaler': scaler,
        'pca': pca,
        'best_model_name': active_model_name,
        'X_test_pca': X_test_pca,  # Save transformed test data for fast CM generation
        'y_test': y_test
    }
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"✅ System saved to '{MODEL_FILENAME}'")

    # 5. Generate Comparison Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
    plt.title('Model Accuracy Comparison')
    plt.savefig('model_comparison.png')

    return True


# ========================================================
# --- STEP 4: LOADING LOGIC ---
# ========================================================

def load_system():
    """
    Attempts to load the system state from the PKL file.
    Returns True if successful, False otherwise.
    """
    global models_pool, scaler, pca, active_model_name, current_model, X_test_pca, y_test

    if not os.path.exists(MODEL_FILENAME):
        print(f"No saved model found at '{MODEL_FILENAME}'")
        return False

    try:
        print(f"\nFound saved system: '{MODEL_FILENAME}'. Loading...")
        with open(MODEL_FILENAME, 'rb') as f:
            data = pickle.load(f)

        models_pool = data['models']
        scaler = data['scaler']
        pca = data['pca']
        active_model_name = data['best_model_name']
        X_test_pca = data.get('X_test_pca')
        y_test = data.get('y_test')

        current_model = models_pool[active_model_name]
        print(f"✅ System loaded successfully! Active Model: {active_model_name}")
        return True
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False


# ========================================================
# --- STEP 5: TESTING FUNCTIONS ---
# ========================================================

def test_single_image(image_path):
    if not os.path.exists(image_path):
        print("Error: Image not found.")
        return

    features = extract_features(image_path)
    if features is None:
        print("Error: Extraction failed.")
        return

    # Transform
    f_scaled = scaler.transform(features.reshape(1, -1))
    f_pca = pca.transform(f_scaled)

    # Predict
    pred = current_model.predict(f_pca)[0]
    prob = current_model.predict_proba(f_pca)[0]

    label = CATEGORIES[pred]
    conf = prob[pred] * 100
    print(f"\n[{active_model_name}] {os.path.basename(image_path)}: {label} ({conf:.2f}%)")


def run_live_test():
    global PROB_THRESHOLD
    print(f"\nLive Test ({active_model_name}) | Threshold: {PROB_THRESHOLD * 100:.0f}% | Press 'q' to quit.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            if roi.size == 0: continue

            roi_res = cv2.resize(clahe.apply(roi), (IMG_SIZE, IMG_SIZE))
            feat = hog(roi_res, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)

            f_pca = pca.transform(scaler.transform(feat))
            probs = current_model.predict_proba(f_pca)[0]
            idx = np.argmax(probs)
            conf = probs[idx]

            if conf >= PROB_THRESHOLD:
                label, color = CATEGORIES[idx], ((0, 255, 0) if idx == 0 else (0, 0, 255))
            else:
                label, color = "Uncertain", (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {conf:.1%}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Live Mask Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()


# ========================================================
# --- STEP 6: MENU ---
# ========================================================

def interactive_menu():
    global active_model_name, current_model, PROB_THRESHOLD

    while True:
        print("\n" + "=" * 60)
        print(f"FACE MASK DETECTION SYSTEM")
        print(f"Active Model: [{active_model_name}] | Threshold: {PROB_THRESHOLD * 100:.0f}%")
        print("=" * 60)
        print("1. Test Custom Image")
        print("2. Test Directory")
        print("3. Live Camera Test")
        print("4. Show Confusion Matrix [Image]")
        print("5. Change Active Model")
        print("6. Change Probability Threshold")
        print("7. RETRAIN MODELS (Full Process)")
        print("8. RELOAD SAVED MODELS (From Disk)")
        print("9. Exit")

        choice = input("\nChoice (1-9): ").strip()

        if choice == '1':
            test_single_image(input("Image path: ").strip())
        elif choice == '2':
            d = input("Directory: ").strip()
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg', '.png')): test_single_image(os.path.join(d, f))
        elif choice == '3':
            run_live_test()
        elif choice == '4':
            if X_test_pca is not None:
                cm = confusion_matrix(y_test, current_model.predict(X_test_pca))
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
                plt.title(f'Confusion Matrix: {active_model_name}')
                plt.show()
            else:
                print("Error: Test data not found. Please Retrain (Opt 7) to generate test data.")
        elif choice == '5':
            print("\nAvailable Models:")
            keys = list(models_pool.keys())
            for i, k in enumerate(keys): print(f"{i + 1}. {k}")
            try:
                active_model_name = keys[int(input("Select #: ")) - 1]
                current_model = models_pool[active_model_name]
                print(f"Switched to {active_model_name}")
            except:
                print("Invalid selection")
        elif choice == '6':
            try:
                PROB_THRESHOLD = float(input("New Threshold (0.5 - 0.99): "))
                print("Updated.")
            except:
                print("Invalid number.")
        elif choice == '7':
            confirm = input("This will overwrite the saved file. Continue? (y/n): ")
            if confirm.lower() == 'y': train_system()
        elif choice == '8':
            load_system()
        elif choice == '9':
            print("Exiting...")
            break


# ========================================================
# --- STEP 7: ENTRY POINT ---
# ========================================================

if __name__ == "__main__":
    # 1. Try to load existing model
    loaded = load_system()

    # 2. If load failed, Force Train
    if not loaded:
        print("Initial setup required.")
        success = train_system()
        if not success:
            print("Failed to initialize system. Exiting.")
            sys.exit()

    # 3. Open Menu
    interactive_menu()