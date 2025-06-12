import { useState, useEffect, useRef } from 'react';
import { FaPhone, FaPhoneSlash } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';

// MULAW encoding/decoding functions
const linearToMulaw = (sample) => {
  const MULAW_BIAS = 132;
  const MULAW_MAX = 32635;
  sample = Math.min(Math.max(sample, -MULAW_MAX), MULAW_MAX);
  const sign = sample < 0 ? 0x80 : 0x00;
  sample = Math.abs(sample);
  const exponent = Math.floor(Math.log2((sample + MULAW_BIAS) / MULAW_BIAS));
  const mantissa = Math.round((sample / Math.pow(2, exponent) - 1) * 16);
  let mulaw = sign | ((exponent & 0x07) << 4) | (mantissa & 0x0F);
  mulaw = mulaw ^ 0xFF; // Invert bits for MULAW
  return mulaw;
};

const mulawToLinear = (mulaw) => {
  const MULAW_BIAS = 132;
  mulaw = mulaw ^ 0xFF; // Invert bits for MULAW
  const sign = mulaw & 0x80 ? -1 : 1;
  const exponent = (mulaw >> 4) & 0x07;
  const mantissa = mulaw & 0x0F;
  const magnitude = (Math.pow(2, exponent) * (16 + mantissa) - 16 + MULAW_BIAS) * Math.pow(2, exponent);
  return sign * magnitude;
};

const Chatbot = ({ isPanelOpen, voiceModel = 'female' }) => {
  const [messages, setMessages] = useState([]);
  const [isInCall, setIsInCall] = useState(false);
  const [websocket, setWebsocket] = useState(null);
  const chatEndRef = useRef(null);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const audioBufferQueue = useRef([]);
  const isPlayingRef = useRef(false);
  const sequenceRef = useRef(0); // Keep incrementing across utterances
  const silenceCountRef = useRef(0);
  const audioChunksRef = useRef({});
  const currentSequence = useRef(0); // Track the next expected sequence for audio playback

  useEffect(() => {
    const timer = setTimeout(() => {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
    return () => clearTimeout(timer);
  }, [messages]);

  useEffect(() => {
    return () => {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.close();
        setWebsocket(null);
      }
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
    };
  }, []);

  const floatTo16BitPCM = (float32Array) => {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      let s = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16Array;
  };

  const calculateRMS = (float32Array) => {
    let sum = 0;
    for (let i = 0; i < float32Array.length; i++) {
      sum += float32Array[i] * float32Array[i];
    }
    return Math.sqrt(sum / float32Array.length);
  };

  const resampleAudio = (inputBuffer, sourceSampleRate, targetSampleRate) => {
    const ratio = sourceSampleRate / targetSampleRate;
    const inputData = inputBuffer.getChannelData(0);
    const outputLength = Math.floor(inputData.length / ratio);
    const outputData = new Float32Array(outputLength);
    for (let i = 0; i < outputLength; i++) {
      const srcIndex = Math.floor(i * ratio);
      outputData[i] = inputData[srcIndex];
    }
    const outputBuffer = audioContextRef.current.createBuffer(1, outputLength, targetSampleRate);
    outputBuffer.getChannelData(0).set(outputData);
    return outputBuffer;
  };

  const playNextBuffer = () => {
    if (audioBufferQueue.current.length === 0 || isPlayingRef.current) return;
    const buffer = audioBufferQueue.current.shift();
    const source = audioContextRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContextRef.current.destination);
    source.onended = () => {
      isPlayingRef.current = false;
      console.log(`[${new Date().toISOString()}] Audio buffer playback ended`);
      playNextBuffer();
    };
    source.start();
    isPlayingRef.current = true;
    console.log(`[${new Date().toISOString()}] Playing audio buffer: duration=${buffer.duration}s`);
  };

  const handleAudioChunk = (sequence, mulawBase64) => {
    try {
      const mulawBytes = atob(mulawBase64);
      const pcmArray = new Int16Array(mulawBytes.length);
      for (let i = 0; i < mulawBytes.length; i++) {
        pcmArray[i] = mulawToLinear(mulawBytes.charCodeAt(i));
      }
      console.log(`Received MULAW chunk: sequence=${sequence}, size=${pcmArray.length} samples`);
      audioChunksRef.current[sequence] = pcmArray;
      while (audioChunksRef.current[currentSequence.current]) {
        const pcm = audioChunksRef.current[currentSequence.current];
        const float32Array = new Float32Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) {
          float32Array[i] = pcm[i] / 32768.0;
        }
        const audioBuffer = audioContextRef.current.createBuffer(1, pcm.length, 8000);
        audioBuffer.getChannelData(0).set(float32Array);
        audioBufferQueue.current.push(audioBuffer);
        delete audioChunksRef.current[currentSequence.current];
        currentSequence.current++;
      }
      playNextBuffer();
    } catch (e) {
      console.error('Error processing audio chunk:', e);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Error processing audio chunk' },
      ]);
    }
  };

  const handleStartCall = async () => {
    if (isInCall) return;
    setIsInCall(true);
    try {
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      console.log('AudioContext initialized at 16000 Hz');

      const ws = new WebSocket('ws://localhost:8000/ws');
      ws.onopen = () => {
        console.log('WebSocket connection established');
        setWebsocket(ws);
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'transcription') {
            setMessages((prev) => [
              ...prev,
              { role: 'user', content: data.text || 'No transcription received' },
            ]);
          } else if (data.type === 'response') {
            setMessages((prev) => [
              ...prev,
              { role: 'assistant', content: data.response || 'No response received' },
            ]);
            currentSequence.current = 0;
            audioChunksRef.current = {};
            audioBufferQueue.current = [];
          } else if (data.type === 'audio_chunk') {
            handleAudioChunk(data.sequence, data.pcm);
          } else if (data.type === 'error') {
            setMessages((prev) => [
              ...prev,
              { role: 'assistant', content: `Error: ${data.message}` },
            ]);
          }
        } catch (e) {
          console.error('Error parsing WebSocket message:', e);
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: 'Error processing message' },
          ]);
        }
      };
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `WebSocket error: ${error.message || 'Connection failed'}` },
        ]);
        setIsInCall(false);
      };
      ws.onclose = () => {
        console.log('WebSocket closed');
        setWebsocket(null);
        setIsInCall(false);
        if (processorRef.current) {
          processorRef.current.disconnect();
          processorRef.current = null;
        }
      };
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
      });
      streamRef.current = stream;
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const processor = audioContextRef.current.createScriptProcessor(2048, 1, 1);
      processorRef.current = processor;
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);
      const silenceThreshold = 0.003;
      const silenceFramesRequired = 100;
      let frameBuffer = [];
      processor.onaudioprocess = (event) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = event.inputBuffer.getChannelData(0);
        const resampledBuffer = resampleAudio(event.inputBuffer, 16000, 8000);
        const resampledData = resampledBuffer.getChannelData(0);
        frameBuffer.push(...resampledData);
        const frameSize = 160;
        while (frameBuffer.length >= frameSize) {
          const frame = frameBuffer.slice(0, frameSize);
          frameBuffer = frameBuffer.slice(frameSize);
          const rms = calculateRMS(new Float32Array(frame));
          const isSilentFrame = rms < silenceThreshold;
          if (isSilentFrame) silenceCountRef.current++;
          else silenceCountRef.current = 0;
          const silenceFlag = silenceCountRef.current >= silenceFramesRequired ? 0 : 1;
          const int16PCM = floatTo16BitPCM(new Float32Array(frame));
          const mulawData = new Uint8Array(int16PCM.length);
          for (let i = 0; i < int16PCM.length; i++) {
            mulawData[i] = linearToMulaw(int16PCM[i]);
          }
          const mulawBase64 = btoa(String.fromCharCode.apply(null, mulawData));
          const message = JSON.stringify({
            client_type: "web",
            sequence: sequenceRef.current,
            silence: silenceFlag,
            pcm: mulawBase64,
          });
          ws.send(message);
          console.log(`[${new Date().toISOString()}] Sent MULAW chunk: sequence=${sequenceRef.current}, silence=${silenceFlag}, RMS=${rms.toFixed(4)}`);
          sequenceRef.current++;
        }
      };
      console.log('Call started, streaming MULAW audio chunks at 8000 Hz');
    } catch (error) {
      console.error('Error starting call:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error accessing microphone: ${error.message}` },
      ]);
      setIsInCall(false);
      if (websocket) websocket.close();
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
    }
  };

  const handleEndCall = () => {
    if (!isInCall) return;
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.close();
    }
    setWebsocket(null);
    setIsInCall(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    sequenceRef.current = 0;
    silenceCountRef.current = 0;
    audioBufferQueue.current = [];
    audioChunksRef.current = {};
    isPlayingRef.current = false;
    console.log('Call ended');
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 bg-gray-800 rounded-lg border border-gray-700">
        <style>
          {`
            .hide-scrollbar::-webkit-scrollbar {
              display: none;
            }
            .hide-scrollbar {
              scrollbar-width: none;
              -ms-overflow-style: none;
            }
          `}
        </style>
        <AnimatePresence>
          {messages.map((msg, index) => (
            <motion.div
              key={index}
              className={`mb-4 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <div
                className={`max-w-[75%] p-3 rounded-lg ${
                  msg.role === 'user' ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-200'
                }`}
                style={{ wordBreak: 'break-word' }}
              >
                {msg.content}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={chatEndRef} />
      </div>
      <div className="mt-4 p-3 bg-gray-800 rounded-lg border border-gray-700">
        <motion.button
          onClick={isInCall ? handleEndCall : handleStartCall}
          className={`w-full p-3 rounded-lg ${
            isInCall ? 'bg-red-600 hover:bg-red-700' : 'bg-cyan-600 hover:bg-cyan-700'
          } text-white flex justify-center items-center`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          aria-label={isInCall ? 'End Call' : 'Start Call'}
        >
          {isInCall ? <FaPhoneSlash className="mr-2" /> : <FaPhone className="mr-2" />}
          {isInCall ? 'End Call' : 'Start Call'}
        </motion.button>
      </div>
    </div>
  );
};

export default Chatbot;