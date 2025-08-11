import cv2
import google.generativeai as genai
import PIL.Image
import io
import os

# Configure Google Gemini API Key
GOOGLE_API_KEY = "API KEY"  #Replace with your Gemini API key
genai.configure(api_key=GOOGLE_API_KEY)

# Resize image to reduce token size
def resize_image(path, max_size=(512, 512)):
    img = PIL.Image.open(path)
    img.thumbnail(max_size)
    img.save(path, format='JPEG', quality=85)

# Convert image to bytes
def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

# Capture image from webcam
def capture_image(filename="captured_image.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Press SPACE to capture an image, or ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow("Webcam - Press SPACE to Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            cv2.imwrite(filename, frame)
            print(f"✅ Image captured and saved as '{filename}'")
            break

    cap.release()
    cv2.destroyAllWindows()
    return filename

# --- Main Logic ---

# Step 1: Capture the image
image_filename = capture_image()
if not image_filename or not os.path.exists(image_filename):
    exit()

# Step 2: Resize the image to reduce input size
resize_image(image_filename)

# Step 3: Load and convert image
image = PIL.Image.open(image_filename)
image_data = image_to_bytes(image)

# Step 4: Ask user for a custom question
user_prompt = input("❓ What do you want to ask Gemini about the image? (e.g. 'Am I smiling?'): ")

# Step 5: Use gemini-1.5-flash model
model = genai.GenerativeModel("gemini-1.5-flash")

# Step 6: Send the image and your prompt to Gemini
response = model.generate_content(
    contents=[
        {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {"text": user_prompt}
            ]
        }
    ]
)

# Step 7: Display the response
print("\n🧠 Gemini AI Response:")

print(response.text)
