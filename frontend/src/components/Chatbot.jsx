import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import "./Chatbot.css";

function Chatbot({ voiceModel, websocket, isConnected, onDisconnect }) {
  const [messages, setMessages] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const userAudioChunksRef = useRef([]);
  const silenceCheckIntervalRef = useRef(null);
  const chunkIntervalRef = useRef(null);
  const isProcessingRef = useRef(false);
  const pcmBufferRef = useRef([]); // Buffer to store PCM data for 1-second chunks
  const sessionIdRef = useRef(0); // Track session for new speech detection

  const SILENCE_THRESHOLD = 0.03;
  const SILENCE_DURATION = 2000; // Fixed 2-second silence duration
  const SAMPLE_RATE = 16000;
  const CHUNK_DURATION_MS = 1000; // 1-second chunks
  const SAMPLES_PER_CHUNK = SAMPLE_RATE * (CHUNK_DURATION_MS / 1000);

  const addMessage = (newMessage) => {
    setMessages((prevMessages) => {
      const updatedMessages = [...prevMessages, newMessage];
      if (updatedMessages.length > 2) {
        return updatedMessages.slice(-2);
      }
      return updatedMessages;
    });
  };

  useEffect(() => {
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: SAMPLE_RATE,
    });

    return () => {
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        audioContextRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (!websocket) {
      console.warn("WebSocket is null, cannot set up event listeners");
      return;
    }

    websocket.onmessage = async (event) => {
      if (typeof event.data === "string") {
        const data = JSON.parse(event.data);
        if (data.type === "transcription") {
          console.log("Received transcription:", data.text);
          addMessage({ type: "user", text: data.text });
          setStatus("Idle");
          setError(null);
          isProcessingRef.current = false;
        } else if (data.type === "response") {
          console.log("Received bot response:", data.text);
          addMessage({ type: "bot", text: data.text });
          setError(null);
        } else if (data.type === "status") {
          console.log("Received status message:", data.message);
          if (data.message === "TTS paused" || data.message === "Bot interrupted") {
            setStatus("Listening");
            isPlayingRef.current = false;
            audioQueueRef.current = [];
          } else if (data.message === "Ready to receive audio") {
            setStatus("Listening");
          }
        } else if (data.type === "error") {
          console.error("WebSocket error:", data.message);
          setError(data.message);
          setStatus("Idle");
          setIsRecording(false);
          isProcessingRef.current = false;
        }
      } else {
        setStatus("Speaking");
        setError(null);
        const audioBlob = event.data;
        try {
          const arrayBuffer = await audioBlob.arrayBuffer();
          const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
          audioQueueRef.current.push(audioBuffer);

          const playNext = async () => {
            if (audioQueueRef.current.length === 0 || !isPlayingRef.current) {
              setStatus(isRecording ? "Listening" : "Idle");
              isPlayingRef.current = false;
              return;
            }
            const buffer = audioQueueRef.current.shift();
            const source = audioContextRef.current.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContextRef.current.destination);
            source.onended = () => {
              playNext();
            };
            source.start();
          };

          if (!isPlayingRef.current) {
            isPlayingRef.current = true;
            playNext();
          }
        } catch (e) {
          console.error("Error playing audio:", e);
          setError("Failed to play bot response");
          isPlayingRef.current = false;
          audioQueueRef.current = [];
        }
      }
    };

    websocket.onclose = () => {
      console.log("WebSocket closed, attempting to reconnect...");
      onDisconnect();
      setIsRecording(false);
      setStatus("Idle");
    };

    return () => {
      websocket.onmessage = null;
      websocket.onclose = null;
    };
  }, [websocket, isRecording, onDisconnect]);

  const sendAudioChunks = () => {
    if (websocket && websocket.readyState === WebSocket.OPEN && userAudioChunksRef.current.length > 0) {
      console.log(`Sending ${userAudioChunksRef.current.length} PCM chunks to backend`);
      userAudioChunksRef.current.forEach((chunk, index) => {
        console.log(`Chunk ${index} size: ${chunk.byteLength} bytes`);
        websocket.send(chunk);
      });
      websocket.send(JSON.stringify({ type: "speech_end" }));
      setStatus("Processing...");
      isProcessingRef.current = true;
    } else {
      console.warn("Cannot send chunks: WebSocket not open or no chunks available");
      if (!websocket) {
        setError("WebSocket connection is not available");
      } else if (websocket.readyState !== WebSocket.OPEN) {
        setError(`WebSocket is not open (state: ${websocket.readyState})`);
      } else {
        setError("No audio chunks to send");
      }
    }
    userAudioChunksRef.current = [];
  };

  const toggleRecording = async () => {
    if (!isConnected || !websocket) {
      console.warn("Cannot toggle recording: WebSocket is not connected or websocket is null");
      setError("WebSocket is not connected. Please try again later.");
      return;
    }

    if (!isRecording) {
      try {
        setStatus("Listening");
        setIsRecording(true);
        setError(null);
        userAudioChunksRef.current = [];
        pcmBufferRef.current = [];
        isProcessingRef.current = false;
        sessionIdRef.current += 1; // Increment session ID for new recording

        if (websocket.readyState === WebSocket.OPEN) {
          console.log("Sending start signal with voice model:", voiceModel);
          websocket.send(JSON.stringify({ type: "start", voice_model: voiceModel }));
        } else {
          console.error("WebSocket is not open, state:", websocket.readyState);
          setError("WebSocket is not open. Please try again.");
          setIsRecording(false);
          setStatus("Idle");
          return;
        }

        // Get audio stream
        try {
          streamRef.current = await navigator.mediaDevices.getUserMedia({
            audio: {
              noiseSuppression: true,
              echoCancellation: true,
              autoGainControl: true,
              sampleRate: SAMPLE_RATE,
              channelCount: 1,
            },
          });
        } catch (err) {
          console.error("Failed to access microphone:", err);
          setError("Failed to access microphone. Please check permissions and try again.");
          setIsRecording(false);
          setStatus("Idle");
          return;
        }

        // Set up audio context and nodes
        const source = audioContextRef.current.createMediaStreamSource(streamRef.current);
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 2048;
        source.connect(analyserRef.current);

        // Use ScriptProcessorNode to capture raw PCM data
        processorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);
        analyserRef.current.connect(processorRef.current);
        processorRef.current.connect(audioContextRef.current.destination);

        const dataArray = new Float32Array(analyserRef.current.fftSize);

        // Capture PCM data
        processorRef.current.onaudioprocess = (event) => {
          const inputData = event.inputBuffer.getChannelData(0);
          pcmBufferRef.current.push(...inputData);
          console.log(`Captured ${inputData.length} PCM samples, total buffer: ${pcmBufferRef.current.length}`);
        };

        // Create PCM chunks every 1 second
        chunkIntervalRef.current = setInterval(() => {
          if (!isRecording) {
            clearInterval(chunkIntervalRef.current);
            return;
          }

          if (pcmBufferRef.current.length >= SAMPLES_PER_CHUNK) {
            const pcmData = new Float32Array(pcmBufferRef.current.slice(0, SAMPLES_PER_CHUNK));
            if (pcmData.length > 0) {
              // Convert Float32Array to Int16Array (16-bit PCM)
              const int16Data = new Int16Array(pcmData.length);
              for (let i = 0; i < pcmData.length; i++) {
                const sample = Math.max(-1, Math.min(1, pcmData[i])) * 0x7FFF; // Scale to 16-bit range
                int16Data[i] = sample;
              }
              const pcmBlob = new Blob([int16Data.buffer], { type: "application/octet-stream" });
              console.log(`Generated PCM chunk, size: ${pcmBlob.size} bytes`);

              if (isProcessingRef.current) {
                console.log("New speech detected while processing, discarding old chunks");
                userAudioChunksRef.current = [];
                isProcessingRef.current = false;
                setStatus("Listening");
                sessionIdRef.current += 1; // New session for new speech
              }
              userAudioChunksRef.current.push(pcmBlob);

              if (isPlayingRef.current && websocket.readyState === WebSocket.OPEN) {
                console.log("User interrupted bot, sending interruption signal");
                websocket.send(JSON.stringify({ type: "user_interrupted" }));
                isPlayingRef.current = false;
                audioQueueRef.current = [];
                setStatus("Listening");
              }
            }
            pcmBufferRef.current = pcmBufferRef.current.slice(SAMPLES_PER_CHUNK);
          }
        }, CHUNK_DURATION_MS);

        // Silence detection with fixed 2-second duration
        silenceCheckIntervalRef.current = setInterval(() => {
          if (!isRecording) {
            clearInterval(silenceCheckIntervalRef.current);
            return;
          }

          analyserRef.current.getFloatTimeDomainData(dataArray);
          const rms = Math.sqrt(
            dataArray.reduce((sum, val) => sum + val * val, 0) / dataArray.length
          );

          if (rms > SILENCE_THRESHOLD) {
            console.log(`Speech detected, RMS: ${rms}`);
            userAudioChunksRef.current.lastSpeechTime = Date.now();
          } else if (userAudioChunksRef.current.length > 0) {
            const now = Date.now();
            const lastSpeechTime = userAudioChunksRef.current.lastSpeechTime || now;
            if (now - lastSpeechTime >= SILENCE_DURATION) {
              console.log("Silence detected for 2 seconds, sending chunks to backend");
              sendAudioChunks();
              userAudioChunksRef.current.lastSpeechTime = null; // Reset after sending
            }
          }
        }, 100);

        console.log("Recording started with 1000ms interval");
      } catch (error) {
        console.error("Error starting recording:", error);
        setError("Failed to start recording: " + error.message);
        setIsRecording(false);
        setStatus("Idle");
      }
    } else {
      setIsRecording(false);
      setStatus("Idle");

      // Clean up intervals
      if (silenceCheckIntervalRef.current) {
        clearInterval(silenceCheckIntervalRef.current);
        silenceCheckIntervalRef.current = null;
      }
      if (chunkIntervalRef.current) {
        clearInterval(chunkIntervalRef.current);
        chunkIntervalRef.current = null;
      }

      // Clean up audio nodes
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      if (analyserRef.current) {
        analyserRef.current.disconnect();
        analyserRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      // Send remaining chunks and close WebSocket
      if (userAudioChunksRef.current.length > 0) {
        sendAudioChunks();
      }
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        console.log("Sending end signal and closing WebSocket");
        websocket.send(JSON.stringify({ type: "end" }));
        websocket.close();
        onDisconnect();
      } else {
        console.warn("WebSocket not available for closing");
        onDisconnect();
      }
    }
  };

  return (
    <div className="chatbot">
      <div className="chat-window">
        {messages.map((msg, index) => (
          <motion.div
            key={index}
            className={`message ${msg.type}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <strong>{msg.type === "user" ? "You" : "Bot"}:</strong> {msg.text}
          </motion.div>
        ))}
        {error && (
          <motion.div
            className="message error"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <strong>Error:</strong> {error}
          </motion.div>
        )}
      </div>
      <div className="controls">
        <button
          onClick={toggleRecording}
          disabled={!isConnected}
          className={isRecording ? "recording" : ""}
        >
          {isRecording ? "Stop Call" : "Start Call"}
        </button>
        <div className="status">Status: {status}</div>
      </div>
    </div>
  );
}

export default Chatbot;