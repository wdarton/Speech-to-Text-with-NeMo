import pyaudio
import numpy as np
import torch
import torchvision  # <-- Add this line here
import tempfile
import os
import wave
import nemo.collections.asr as nemo_asr


# ==========================================
# Configuration & Hyperparameters
# ==========================================
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.63  # MarbleNet works well with 0.63s windows
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
FORMAT = pyaudio.paInt16
CHANNELS = 1

VAD_THRESHOLD = 0.6      # Probability threshold for speech
SILENCE_PATIENCE = 2     # Number of consecutive silent chunks to trigger transcription

# ==========================================
# 1. Load NeMo Models
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading MarbleNet VAD model...")
# Using the multilingual marblenet which is robust to background noise
vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
    model_name="vad_multilingual_marblenet"
).to(device)
vad_model.eval()

print("Loading Parakeet-TDT-0.6b-v2 ASR model...")
# 600M parameter model with punctuation and capitalization
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
).to(device)
asr_model.eval()

# ==========================================
# 2. Helper Functions
# ==========================================
def is_speech(audio_float32):
    """Passes a chunk of audio through MarbleNet to determine if it contains speech."""
    # Reshape for NeMo: (batch_size, sequence_length)
    signal = torch.tensor(audio_float32, dtype=torch.float32).unsqueeze(0).to(device)
    signal_length = torch.tensor([signal.shape[1]], dtype=torch.int32).to(device)
    
    with torch.no_grad():
        # Forward pass through VAD model
        log_probs = vad_model.forward(input_signal=signal, input_signal_length=signal_length)
        probs = torch.softmax(log_probs, dim=-1)
        
        # MarbleNet classes: index 0 = background, index 1 = speech
        speech_prob = probs[0, 1].item()
        
    return speech_prob > VAD_THRESHOLD

def transcribe_buffer(audio_buffer):
    """Saves the buffered audio to a temporary file and transcribes it with Parakeet."""
    # Convert float32 buffer back to int16 for WAV saving
    audio_int16 = (np.concatenate(audio_buffer) * 32767).astype(np.int16)
    
    # Save to temp file
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    
    # Transcribe
    with torch.no_grad():
        raw_output = asr_model.transcribe([temp_path])
        
    os.remove(temp_path)
    
    # --- NEW EXTRACTION LOGIC ---
    # 1. If NeMo returned a tuple (common for TDT/RNNT models), grab the first element
    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]
        
    # 2. Grab the first item from the batch list (since we only sent one audio file)
    if isinstance(raw_output, list) and len(raw_output) > 0:
        transcription = raw_output[0]
    else:
        transcription = raw_output

    # 3. If the item is a NeMo Hypothesis object, extract its .text attribute
    if hasattr(transcription, 'text'):
        final_text = transcription.text
    # 4. Otherwise, if it's already a string, keep it
    elif isinstance(transcription, str):
        final_text = transcription
    else:
        final_text = str(transcription)
        
    return final_text

# ==========================================
# 3. Real-Time Audio Loop
# ==========================================
def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SAMPLES)

    print("\n" + "="*50)
    print("Listening... (Press Ctrl+C to stop)")
    print("="*50 + "\n")

    audio_buffer = []
    silence_counter = 0
    is_recording_speech = False

    try:
        while True:
            # 1. Read raw bytes from microphone
            raw_data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            
            # 2. Convert to float32 numpy array [-1.0, 1.0] for the model
            chunk_int16 = np.frombuffer(raw_data, dtype=np.int16)
            chunk_float32 = chunk_int16.astype(np.float32) / 32768.0
            
            # 3. Check for speech using MarbleNet
            speech_detected = is_speech(chunk_float32)

            if speech_detected:
                if not is_recording_speech:
                    print("\n[Speech Detected] Recording...", end="", flush=True)
                else:
                    print(".", end="", flush=True)
                    
                is_recording_speech = True
                audio_buffer.append(chunk_float32)
                silence_counter = 0  # Reset silence counter
                
            elif is_recording_speech:
                # Speech has paused/stopped, increment silence counter
                silence_counter += 1
                audio_buffer.append(chunk_float32) # Keep some trailing silence
                
                # If silence duration exceeds our patience, transcribe!
                if silence_counter >= SILENCE_PATIENCE:
                    print("\n[Silence Detected] Transcribing...")
                    
                    text = transcribe_buffer(audio_buffer)
                    print(f"--> Parakeet: {text}\n")
                    
                    # Reset state for the next utterance
                    audio_buffer = []
                    is_recording_speech = False
                    silence_counter = 0

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
