
# Cattle Count Project Using YOLOv9 and WebRTC

Welcome to the Cattle Count Project! This project aims to count cattle using the YOLOv9 object detection model, with real-time video streaming and communication facilitated by WebRTC. The project consists of a server-client architecture where the server processes video frames for cattle detection and the client streams the video and receives the detection results.


<div align="center">
    <img src="video.gif" alt="Cattle Count Demo">
</div>


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The Cattle Count Project is designed to provide a robust and efficient method for counting cattle in real-time using state-of-the-art object detection technology. By leveraging the YOLOv9 model and WebRTC for video streaming, this project ensures accurate and fast cattle detection, making it suitable for various agricultural and farming applications.

## Features
- **Real-time Cattle Detection**: Utilizes the YOLOv9 model for accurate and efficient cattle detection.
- **WebRTC Integration**: Enables real-time video streaming and communication between the server and client.
- **Scalable Architecture**: Server-client architecture allows for easy scaling and deployment.
- **User-Friendly Interface**: Simple and intuitive interface for easy interaction and monitoring.

## Architecture
The project follows a server-client architecture, where:
- **Server**: Handles the processing of video frames, applies the YOLOv9 model for cattle detection, and sends the detection results back to the client.
- **Client**: Streams video to the server and displays the detection results received from the server in real-time.

### Communication via WebRTC
WebRTC (Web Real-Time Communication) is used to facilitate the real-time video streaming and communication between the server and client. This ensures low-latency and high-quality video transmission, which is crucial for real-time cattle detection.

## Installation
To set up and run the project, follow these steps:

### Prerequisites
- Python 3.11
- Node.js and npm
- OpenCV
- YOLOv9 pre-trained weights
- WebRTC libraries

### Steps

1. **Create and activate a Conda environment**:
    ```bash
    conda create --name cattle_count python=3.11
    conda activate cattle_count
    ```

2. **Install server dependencies**:
    ```bash
    cd yolov9
    pip install -r requirements.txt
    cd ..
    pip install -r requirements.txt
    ```

3. **Install client dependencies**:
    ```bash
    cd client
    npm install
    ```

4. **Download YOLOv9 weights**:
    Download the pre-trained YOLOv9 weights and place them in the `server/models` directory.

### Deactivating the Conda environment (when done):
    ```bash
    conda deactivate
    ```

## Usage
To run the project, follow these steps:

### Start the Server
1. Navigate to the server directory:
    ```bash
    cd server
    ```

2. Run the start script:
    ```bash
    python start.py
    ```

### Start the Client
1. Navigate to the client directory:
    ```bash
    cd client
    ```

2. Start the client application:
    ```bash
    npm start
    ```

3. Open your web browser and go to `http://localhost:3000` to access the client interface.

## Contributing
Contributions are welcome! If you would like to contribute to the project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Thank you for using the Cattle Count Project! We hope this project helps you achieve accurate and efficient cattle counting using the power of YOLOv9 and WebRTC. If you have any questions or need further assistance, please feel free to open an issue or contact us.
