# Face-mask-Detection 😷
 
**Objective:🎯**

To develop a real-time face mask detection system using Python, Keras, OpenCV, and NumPy, capable of accurately identifying whether individuals in images or video streams are wearing face masks. This system can contribute to public health efforts by promoting mask compliance in various settings.

**Libraries and Tools:🔨**

- Python: Primary programming language
- Keras: Deep learning framework for model development
- OpenCV (cv2): Image and video processing library
- NumPy: Numerical computation library for data manipulation

**Key Modules:🗝️**

1. **Data Collection and Preprocessing:**
   - Gather a diverse dataset of images containing faces with and without masks.
   - Preprocess images through resizing, normalization, and augmentation (if needed).
   - Split data into training and validation sets.

2. **Model Development:**
   - Design a convolutional neural network (CNN) architecture using Keras.
   - Configure model layers (e.g., convolutional, pooling, dense), activation functions, and optimizer.
   - Train the model on the prepared dataset to learn mask-related features.

3. **Face Detection:**
   - Utilize OpenCV's Haar cascade classifier or deep learning-based face detectors for real-time face detection.

4. **Mask Prediction:**
   - For each detected face:
     - Extract the face region of interest (ROI).
     - Preprocess the ROI to match model input requirements.
     - Feed the ROI to the trained model to obtain mask/no-mask prediction.

5. **Output Visualization:**
   - Display results on images or video frames:
     - Draw bounding boxes around detected faces.
     - Overlay text indicating mask presence or absence.
     - Explore options for visual feedback (e.g., colors, icons).

**Additional Considerations:**

- **Deployment:** Consider deployment strategies for real-world applications (e.g., web applications, integrated systems).
- **Optimization:** Explore techniques to enhance performance and accuracy, such as model compression or hardware acceleration.
- **Ethical Considerations:** Address privacy concerns and ensure responsible use of the system.

**Potential Applications:**

- Public spaces (e.g., transportation, retail stores, offices)
- Healthcare facilities
- Security systems
- Personal safety devices
- Research and educational purposes

**Working 🛠️** 

Project Components:🔩⚙️

1. **Model Loading:**
   - The project begins by loading a pre-trained deep learning model using Keras. The model is saved in the "keras_Model.h5" file, and it is loaded without recompiling to ensure quick and efficient predictions.

```python
model = load_model("keras_Model.h5", compile=False)
```

2. **Label Loading:🏷️**
   - Class labels are loaded from the "labels.txt" file. These labels are associated with the classes predicted by the model.

```python
class_names = open("labels.txt", "r").readlines()
```

3. **Webcam Integration:🎦**
   - The OpenCV library is used to access the webcam (Camera 0 by default) and capture real-time video frames.

```python
camera = cv2.VideoCapture(0)
```

4. **Real-time Prediction Loop:♾️**
   - The program enters a continuous loop, capturing webcam frames, resizing them to the required input size (224x224 pixels), and displaying the frames in a window.

```python
while True:
    # ... (Image processing and display)
```

5. **Image Preprocessing:👨🏼‍💻**
   - Each captured frame is converted into a numpy array, reshaped to match the input shape expected by the model, and normalized to be within the range [-1, 1].

```python
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
image = (image / 127.5) - 1
```

6. **Prediction and Display:👨🏼‍💻**
   - The pre-trained model is used to predict the class of the object in the frame, and the result is displayed along with a confidence score.

```python
prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]
print("Class:", class_name[2:], end="")
print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
```

7. **User Interaction:👨🏼‍💻**
   - The application listens for keyboard input, and if the 'Esc' key (ASCII code 27) is pressed, the program exits the loop, releasing the camera and closing the OpenCV windows.

```python
keyboard_input = cv2.waitKey(1)
if keyboard_input == 27:
    break
```

8. **Cleanup:🧹**
   - After the loop is terminated, the camera is released, and OpenCV windows are destroyed.

```python
camera.release()
cv2.destroyAllWindows()
```

### Conclusion:
This real-time object classification project combines the strengths of deep learning and computer vision, providing a versatile tool for quick and accurate object recognition through a webcam feed. Users can easily extend and customize the project by training their models or adapting the existing code for specific use cases.
