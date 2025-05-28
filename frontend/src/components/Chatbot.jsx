import { useState, useEffect, useRef } from 'react';
import { FaPhone, FaPhoneSlash } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';

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

  const handleAudioChunk = (sequence, pcmBase64) => {
    try {
      const pcmBytes = atob(pcmBase64);
      const pcmArray = new Int16Array(pcmBytes.length / 2);
      for (let i = 0; i < pcmBytes.length; i += 2) {
        pcmArray[i / 2] = (pcmBytes.charCodeAt(i) & 0xff) | (pcmBytes.charCodeAt(i + 1) << 8);
      }
      console.log(`Received chunk: sequence=${sequence}, size=${pcmArray.length} samples`);
      audioChunksRef.current[sequence] = pcmArray;
      // Process chunks in order starting from currentSequence
      while (audioChunksRef.current[currentSequence.current]) {
        const pcm = audioChunksRef.current[currentSequence.current];
        const float32Array = new Float32Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) {
          float32Array[i] = pcm[i] / 32768.0;
        }
        const audioBuffer = audioContextRef.current.createBuffer(1, pcm.length, 16000);
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
      console.log('AudioContext initialized');

      const ws = new WebSocket('ws://localhost:8000/ws/chat');
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
            // Reset currentSequence for new response
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
      const processor = audioContextRef.current.createScriptProcessor(8192, 1, 1);
      processorRef.current = processor;
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);
      const silenceThreshold = 0.005;
      const silenceChunksRequired = 2; // ~1s silence
      processor.onaudioprocess = (event) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = event.inputBuffer.getChannelData(0);
        const rms = calculateRMS(input);
        const isSilentChunk = rms < silenceThreshold;
        if (isSilentChunk) silenceCountRef.current++;
        else silenceCountRef.current = 0;
        const silenceFlag = silenceCountRef.current >= silenceChunksRequired ? 0 : 1;
        const int16PCM = floatTo16BitPCM(new Float32Array(input));
        const pcmBase64 = btoa(String.fromCharCode.apply(null, new Uint8Array(int16PCM.buffer)));
        const message = JSON.stringify({
          sequence: sequenceRef.current,
          silence: silenceFlag,
          pcm: pcmBase64,
        });
        ws.send(message);
        console.log(`[${new Date().toISOString()}] Sent chunk: sequence=${sequenceRef.current}, silence=${silenceFlag}`);
        sequenceRef.current++; // Increment sequence number for each chunk
      };
      console.log('Call started, streaming audio chunks');
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