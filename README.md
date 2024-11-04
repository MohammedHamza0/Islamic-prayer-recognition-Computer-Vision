# Islamic Prayer Pose Detection with YOLO

This project utilizes YOLO (You Only Look Once) object detection and computer vision techniques to detect and classify various Islamic prayer poses in real-time. It aims to identify key prayer postures like "Ruku" (bowing), "Sujud" (prostration), "Takbeer" (raising hands), and more. Additionally, the project provides real-time feedback by counting the number of "Ruku" instances and displaying text in Arabic over the detected poses using OpenCV and PIL.

## Features

- **Real-time Islamic Prayer Pose Detection**: Identifies key prayer postures using the YOLO algorithm.
- **Pose Classification**: The model classifies the following poses:
  - Raising (قائم)
  - Bowing (راكع)
  - Sujud (ساجد)
  - Tashahhud (التشهد)
  - Takbeer (تكبير)
- **Dynamic Arabic Text Overlay**: Displays posture names in Arabic, reshaped and rendered using `arabic-reshaper` and `PIL` for accurate display.
- **Ruku Count**: The system automatically tracks and counts the number of "Ruku" (bowing) positions held for a specified time.
- **Customizable Display**: The project allows text color, background, and border styles to be customized for enhanced visibility.

## Dependencies

The project requires the following libraries:

- `opencv-python`: For handling video frames and drawing bounding boxes.
- `ultralytics`: YOLO object detection package.
- `bidi.algorithm`: To handle bidirectional display of Arabic text.
- `arabic-reshaper`: To reshape Arabic text for proper display.
- `Pillow (PIL)`: For drawing text over frames.
- `numpy`: For handling arrays and data manipulation.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MohammedHamza0/IslamicPrayerPoseDetection.git
   cd IslamicPrayerPoseDetection
   ```

2. **Install dependencies**:
   Ensure you have Python 3.7 or higher. Install the necessary packages by running:
   ```bash
   pip install opencv-python ultralytics bidi.algorithm arabic-reshaper Pillow numpy
   ```

3. **Model Weights**:
   Download the `IslamicBest.pt` model weights from the appropriate source and place it in the project directory.

4. **Running the Program**:
   Run the script:
   ```bash
   python islamic_prayer_recognition.py
   ```

   Replace `islamic_prayer_recognition.py` with the script name if different.

## How It Works

1. **Video Input**: The program reads a video stream using OpenCV. In the provided example, it processes the video file `istockphoto-1345393460-640_adpp_is.mp4`.
2. **Pose Detection**: YOLO is used to predict bounding boxes around people in different prayer postures.
3. **Arabic Text Display**: The corresponding Arabic text for each pose is drawn over the detected individual using `PIL` and reshaped for proper display with `arabic-reshaper`.
4. **Ruku Counting**: The system tracks the duration of the "Ruku" position and increments a counter if the bowing position is held for a defined duration (e.g., 10 seconds).

## Customization

- **Text Settings**: You can adjust the font type, size, text color, and background by modifying the `draw_text_with_background` function.
- **Detection Threshold**: Adjust the confidence threshold in the `model.predict` call to fine-tune detection accuracy:
   ```python
   results = model.predict(frame, conf=0.35)
   ```

## Usage

- **Real-time Detection**: Detects Islamic prayer postures in real-time and displays their Arabic names on the video feed.
- **Ruku Counter**: Automatically tracks and counts the number of "Ruku" positions during the prayer.

## Example Output

The project displays the following on the video stream:
- Bounding boxes around people performing different prayer poses.
- Arabic text indicating the current prayer pose.
- A counter showing the number of "Ruku" (bowing) completed.

## Acknowledgments

- **YOLO**: For object detection.
- **OpenCV**: For image and video processing.
- **PIL (Pillow)**: For text rendering on images.
- **bidi.algorithm** and **arabic-reshaper**: For proper display of Arabic text.
