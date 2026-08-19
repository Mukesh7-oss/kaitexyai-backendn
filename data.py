import os, zipfile, shutil, cv2, numpy as np, mediapipe as mp
from multiprocessing import Pool, cpu_count

EXTRACT_DIR = "temp_dataset"
OUTPUT_DIR = "landmark_dataset"
ZIP_PATH = "D:/KaitexyAI-new/backend/asl_alphabet_train.zip"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def clean_dirs():
    for folder in [EXTRACT_DIR, OUTPUT_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

def extract_zip():
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

def find_dataset_root(path):
    for root, dirs, files in os.walk(path):
        image_class_count = 0
        for d in dirs:
            subdir = os.path.join(root, d)
            try:
                if any(f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(subdir)):
                    image_class_count += 1
            except Exception:
                pass
        if image_class_count >= 2:
            return root
    return None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def process_image(args):
    image_path, output_class_path = args
    image = cv2.imread(image_path)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    landmarks = []
    for lm in results.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    if len(landmarks) != 63:
        return None
    out_file = os.path.join(output_class_path, os.path.splitext(os.path.basename(image_path))[0] + ".npy")
    np.save(out_file, np.array(landmarks, dtype=np.float32))
    return out_file

if __name__ == "__main__":
    clean_dirs()
    extract_zip()
    dataset_path = find_dataset_root(EXTRACT_DIR)
    class_folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])

    tasks = []
    for cls in class_folders:
        in_path = os.path.join(dataset_path, cls)
        out_path = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(out_path, exist_ok=True)
        for f in os.listdir(in_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                tasks.append((os.path.join(in_path, f), out_path))

    print(f"Processing {len(tasks)} images with {cpu_count()} cores...")
    with Pool(cpu_count()) as p:
        results = p.map(process_image, tasks)

    print("Done. Extracted landmarks:", sum(1 for r in results if r))
