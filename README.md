# **Real-Time Speech-to-Text with NVIDIA NeMo**

This guide provides setup instructions for running the real-time Voice Activity Detection (VAD) and Speech-to-Text (ASR) pipeline on a **Debian-based system** (such as Debian 11/12 or Ubuntu).

The pipeline uses **MarbleNet** to detect speech chunks and **Parakeet-TDT-0.6b-v2** to transcribe them on the fly.

## **Prerequisites (Debian)**

The script requires a CUDA-enabled NVIDIA GPU to run in real-time, as well as specific system packages for audio processing. If an NVIDIA GPU is not located, it will fall back to CPU. 

### **1. System Dependencies**

Open a terminal and install the required Debian packages. portaudio19-dev is strictly required to successfully compile and install the pyaudio Python library.
````
sudo apt-get update  
sudo apt-get install -y portaudio19-dev python3-dev python3-pip python3-venv build-essential
````
### **2. Python Environment Setup**

It is highly recommended to use a Python virtual environment to avoid conflicting with system-level Python packages on Debian.
````
# Create a virtual environment named 'asr_env'  
python3 -m venv asr_env

# Activate the virtual environment  
source asr_env/bin/activate
````
### **3. Install Python Dependencies**

First, install PyTorch, Torchvision, and Torchaudio with CUDA support. This setup uses the CUDA 12.1 builds, which are highly stable for modern NVIDIA cards.
````
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
````
Next, install the NVIDIA NeMo toolkit and necessary audio processing libraries:
````
pip install "nemo_toolkit[asr]" pyaudio numpy scipy
````
## **Usage**

1. Save the provided Python script as realtime_asr.py in your working directory.  
2. Ensure your microphone is connected and configured as the default input device on your Debian system.  
3. Run the script:
````
python realtime_asr.py
````
### **How it Works**

* The script listens to your microphone in continuous 0.63-second chunks.  
* **MarbleNet** evaluates each chunk. If speech is detected, the script starts buffering.  
* Once the speaker pauses (configurable via SILENCE_PATIENCE in the script), the buffered audio is packaged and sent to **Parakeet-TDT-0.6b-v2**.  
* The clean, transcribed text string is parsed and printed to your console.

## **Troubleshooting**

* **RuntimeError: operator torchvision::nms does not exist**: The provided script already handles this by explicitly including import torchvision at the top of the file to force C++ extensions to initialize before NeMo is called. Ensure this line remains intact.  
* **ALSA lib warnings on startup**: You may see a large block of ALSA lib pcm.c... and jack server is not running warnings when PyAudio initializes. This is standard behavior on Debian/Linux ALSA audio subsystems. It is not an error and can be safely ignored as long as the microphone is successfully captured and the Listening... prompt appears.  
* **Microphone Permission Denied**: If you get a permission error accessing the audio device, ensure your user is added to the audio group: `sudo usermod -aG audio $USER`, then log out and log back in.
