import { useState, useEffect, useRef } from 'react';
import { FaPhone, FaPhoneSlash, FaPlayCircle, FaPauseCircle } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';

const Chatbot = ({ isPanelOpen, voiceModel = 'female' }) => {
  const [messages, setMessages] = useState([]);
  const [isInCall, setIsInCall] = useState(false);
  const [websocket, setWebsocket] = useState(null);
  const chatEndRef = useRef(null);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const pcmBufferRef = useRef([]);
  const silenceStartRef = useRef(null);
  const currentAudioRef = useRef(null);
  const playingMessageIdRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
    return () => clearTimeout(timer);
  }, [messages]);

  useEffect(() => {
    return () => {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
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

  const isSilent = (float32Array) => {
    const threshold = 0.02; // Adjust for sensitivity
    for (let i = 0; i < float32Array.length; i++) {
      if (Math.abs(float32Array[i]) > threshold) {
        return false;
      }
    }
    return true;
  };

  const handlePlayAudio = (audioUrl, messageId) => {
    if (!audioUrl) return;

    const fullAudioUrl = audioUrl.startsWith('http') ? audioUrl : `http://localhost:8000${audioUrl}`;
    if (playingMessageIdRef.current === messageId && currentAudioRef.current) {
      if (isPlaying) {
        currentAudioRef.current.pause();
        setIsPlaying(false);
      } else {
        currentAudioRef.current.play().catch((e) => {
          console.error('Audio playback failed:', e);
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: 'Error playing audio: ' + e.message },
          ]);
        });
        setIsPlaying(true);
      }
      return;
    }

    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }

    const audio = new Audio(fullAudioUrl);
    currentAudioRef.current = audio;
    playingMessageIdRef.current = messageId;
    setIsPlaying(true);

    audio.oncanplay = () => {
      audio.play().catch((e) => {
        console.error('Audio playback failed:', e);
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: 'Error playing audio: ' + e.message },
        ]);
        currentAudioRef.current = null;
        playingMessageIdRef.current = null;
        setIsPlaying(false);
      });
    };

    audio.onended = () => {
      currentAudioRef.current = null;
      playingMessageIdRef.current = null;
      setIsPlaying(false);
    };

    audio.onerror = () => {
      console.error('Audio loading error');
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Error loading audio' },
      ]);
      currentAudioRef.current = null;
      playingMessageIdRef.current = null;
      setIsPlaying(false);
    };

    audio.load();
  };

  const handleStartCall = async () => {
    if (isInCall) return;

    setIsInCall(true);
    const ws = new WebSocket('ws://localhost:8000/ws/chat');
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      console.log('WebSocket connection established');
      setWebsocket(ws);
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'response') {
          const botMessage = {
            role: 'assistant',
            content: data.response || 'No response received.',
            audioUrl: data.audio_url,
          };
          setMessages((prev) => [...prev, botMessage]);
        } else if (data.type === 'error') {
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: 'Error: ' + data.message },
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
        { role: 'assistant', content: 'WebSocket error' },
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

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
      });
      streamRef.current = stream;
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      source.connect(processor);
      processor.connect(audioContextRef.current.destination);

      processor.onaudioprocess = (event) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          return;
        }
        const input = event.inputBuffer.getChannelData(0);
        const int16PCM = floatTo16BitPCM(new Float32Array(input));
        pcmBufferRef.current.push(int16PCM);

        if (isSilent(input)) {
          if (!silenceStartRef.current) {
            silenceStartRef.current = Date.now();
          } else if (Date.now() - silenceStartRef.current >= 2000) {
            if (pcmBufferRef.current.length > 0) {
              const totalSamples = pcmBufferRef.current.reduce((sum, chunk) => sum + chunk.length, 0);
              const combinedPCM = new Int16Array(totalSamples);
              let offset = 0;
              pcmBufferRef.current.forEach((chunk) => {
                combinedPCM.set(chunk, offset);
                offset += chunk.length;
              });
              ws.send(combinedPCM.buffer);
              console.log(`[${new Date().toISOString()}] Sent PCM: ${totalSamples} samples`);
              pcmBufferRef.current = [];
            }
            silenceStartRef.current = null;
          }
        } else {
          silenceStartRef.current = null;
        }
      };

      console.log('Call started, streaming PCM audio');
    } catch (error) {
      console.error('Error starting call:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Error accessing microphone: ' + error.message },
      ]);
      setIsInCall(false);
      if (ws) ws.close();
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
    pcmBufferRef.current = [];
    silenceStartRef.current = null;
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
                {msg.audioUrl && msg.role === 'assistant' && (
                  <motion.button
                    onClick={() => handlePlayAudio(msg.audioUrl, index)}
                    className="mt-2 flex items-center text-cyan-400 hover:text-cyan-300"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    aria-label={playingMessageIdRef.current === index && isPlaying ? 'Pause Audio' : 'Play Audio'}
                  >
                    {playingMessageIdRef.current === index && isPlaying ? (
                      <FaPauseCircle className="mr-1" />
                    ) : (
                      <FaPlayCircle className="mr-1" />
                    )}
                    {playingMessageIdRef.current === index && isPlaying ? 'Pause' : 'Play'}
                  </motion.button>
                )}
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