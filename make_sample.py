"""Creates a printed-style sample image so you can smoke-test the pipeline without a camera."""
import cv2, numpy as np
img = np.full((360, 900, 3), 235, np.uint8)
for i, t in enumerate(["Meeting notes - 12 Sept", "Buy milk, eggs and bread", "Call Ravi about the project demo"]):
    cv2.putText(img, t, (30, 90 + i * 90), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (40, 40, 40), 2, cv2.LINE_AA)
cv2.imwrite("samples/sample_note.png", img)
