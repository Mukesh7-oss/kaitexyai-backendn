import cv2
import mediapipe as mp
import numpy as np
import os


label = "Z"
samples_to_collect = 2000
save_dir = f"dataset/{label}"

# Create folder
os.makedirs(save_dir, exist_ok=True)


# ========== MEDIAPIPE ==========

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ========== CAMERA ==========

cap = cv2.VideoCapture(0)

count = 0

print(f"Collecting data for: {label}")
print("Press 'S' to START / RESUME collecting")
print("Press 'P' to PAUSE collecting")
print("Press 'Q' to QUIT")


# Collection state
collecting = False


# ========== MAIN LOOP ==========

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip camera horizontally
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe processing
    result = hands.process(rgb)


    # ============================================================
    # COLLECT LANDMARKS
    # ============================================================

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            points = []

            for lm in hand_landmarks.landmark:

                points.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])


            # Save only when collection is active
            if collecting and count < samples_to_collect:

                np.save(
                    f"{save_dir}/{count}.npy",
                    points
                )

                count += 1


    # ============================================================
    # DISPLAY INFORMATION
    # ============================================================

    cv2.putText(
        frame,
        f"Label: {label}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )


    cv2.putText(
        frame,
        f"Saved: {count}/{samples_to_collect}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ============================================================
    # STATUS
    # ============================================================

    if collecting:

        cv2.putText(
            frame,
            "STATUS: COLLECTING",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "STATUS: PAUSED",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    # ============================================================
    # KEYBOARD INSTRUCTIONS
    # ============================================================

    cv2.putText(
        frame,
        "S = Start/Resume | P = Pause | Q = Quit",
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # Display camera
    cv2.imshow(
        "Dataset Collector",
        frame
    )


    # ============================================================
    # KEYBOARD INPUT
    # ============================================================

    key = cv2.waitKey(1) & 0xFF


    # START / RESUME
    if key == ord('s') or key == ord('S'):

        collecting = True

        print("Collection STARTED / RESUMED")


    # PAUSE
    elif key == ord('p') or key == ord('P'):

        collecting = False

        print(f"Collection PAUSED at {count}/{samples_to_collect}")


    # QUIT
    elif key == ord('q') or key == ord('Q'):

        print(f"Collection stopped at {count}/{samples_to_collect}")

        break


    # ============================================================
    # AUTOMATIC FINISH
    # ============================================================

    if count >= samples_to_collect:

        print(
            f"Finished collecting "
            f"{samples_to_collect} samples for {label}"
        )

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()