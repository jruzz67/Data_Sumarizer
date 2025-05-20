import { useState, useEffect, useRef } from 'react';
import { FaMicrophone, FaPaperPlane, FaStopCircle, FaSpinner, FaPlayCircle, FaPauseCircle } from 'react-icons/fa';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

const Chatbot = ({ isPanelOpen, voiceModel = 'female' }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const chatEndRef = useRef(null);
  const [isSending, setIsSending] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isBotTyping, setIsBotTyping] = useState(false);
  const [currentAudio, setCurrentAudio] = useState(null);
  const [playingMessageId, setPlayingMessageId] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
    return () => clearTimeout(timer);
  }, [messages]);

  useEffect(() => {
    return () => {
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        setCurrentAudio(null);
        setPlayingMessageId(null);
        setIsPlaying(false);
      }
    };
  }, [currentAudio]);

  const stopCurrentAudio = () => {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.currentTime = 0;
      setCurrentAudio(null);
      setPlayingMessageId(null);
      setIsPlaying(false);
    }
  };

  const handleSendMessage = async (queryText) => {
    if (!queryText.trim() || isSending) return;

    setIsSending(true);
    const userMessage = { role: 'user', content: queryText };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsBotTyping(true);

    try {
      console.log('Sending query with voice model:', voiceModel);
      const response = await axios.post('http://localhost:8000/query', { query: queryText, voice_model: voiceModel });
      const botContent = response.data?.response || 'No response content received.';
      const audioUrl = response.data?.audio_url;

      const botMessage = { role: 'assistant', content: botContent, audioUrl };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Error: ' + (error.response?.data?.detail || error.message || 'An unexpected error occurred.'),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsSending(false);
      setIsBotTyping(false);
    }
  };

  const handlePlayAudio = (audioUrl, messageId) => {
    if (!audioUrl) return;

    const fullAudioUrl = audioUrl.startsWith('http') ? audioUrl : `http://localhost:8000${audioUrl}`;
    console.log('Handling audio for:', fullAudioUrl);

    if (playingMessageId === messageId && currentAudio) {
      if (isPlaying) {
        currentAudio.pause();
        setIsPlaying(false);
      } else {
        currentAudio.play().catch((e) => {
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

    stopCurrentAudio();

    const audio = new Audio(fullAudioUrl);
    setCurrentAudio(audio);
    setPlayingMessageId(messageId);
    setIsPlaying(true);

    audio.oncanplay = () => {
      console.log('Audio is ready to play');
      audio.play().catch((e) => {
        console.error('Audio playback failed:', e);
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: 'Error playing audio: ' + e.message },
        ]);
        setCurrentAudio(null);
        setPlayingMessageId(null);
        setIsPlaying(false);
      });
    };

    audio.onended = () => {
      setCurrentAudio(null);
      setPlayingMessageId(null);
      setIsPlaying(false);
    };

    audio.onerror = (e) => {
      console.error('Audio loading error:', e);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Error loading audio: ' + e.message },
      ]);
      setCurrentAudio(null);
      setPlayingMessageId(null);
      setIsPlaying(false);
    };

    audio.load();
  };

  const handleRecord = async () => {
    stopCurrentAudio();

    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const options = { mimeType: 'audio/webm;codecs=opus' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
          options.mimeType = 'audio/webm';
          if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            console.error('No supported audio MIME type found for MediaRecorder.');
            setMessages((prev) => [...prev, { role: 'assistant', content: 'Error: Your browser does not support audio recording.' }]);
            return;
          }
        }
        console.log('Using MIME type:', options.mimeType);

        const recorder = new MediaRecorder(stream, options);
        const chunks = [];

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunks.push(e.data);
            console.log('Chunk received:', e.data.size);
          }
        };

        recorder.onstop = async () => {
          console.log('Recording stopped, total chunks size:', chunks.reduce((acc, chunk) => acc + chunk.size, 0));
          setIsRecording(false);
          setIsTranscribing(true);

          if (chunks.length === 0 || chunks.reduce((acc, chunk) => acc + chunk.size, 0) === 0) {
            setMessages((prev) => [...prev, { role: 'assistant', content: 'Error: No valid audio data recorded.' }]);
            setIsTranscribing(false);
            return;
          }

          const audioBlob = new Blob(chunks, { type: options.mimeType });
          console.log('Audio Blob:', audioBlob, 'Size:', audioBlob.size);

          const formData = new FormData();
          formData.append('file', audioBlob, 'recording.webm');

          try {
            const response = await axios.post('http://localhost:8000/transcribe', formData);
            const transcribedText = response.data?.transcribed_text;

            if (!transcribedText) {
              setMessages((prev) => [...prev, { role: 'assistant', content: 'Transcription failed: Received empty text.' }]);
              return;
            }

            setInput(transcribedText);
            handleSendMessage(transcribedText);
          } catch (error) {
            console.error('Error transcribing audio:', error);
            setMessages((prev) => [
              ...prev,
              { role: 'assistant', content: 'Error transcribing audio: ' + (error.response?.data?.detail || error.message || 'An unexpected error occurred.') },
            ]);
          } finally {
            setIsTranscribing(false);
          }
        };

        recorder.onerror = (e) => {
          console.error('MediaRecorder error:', e);
          setMessages((prev) => [...prev, { role: 'assistant', content: 'Error recording audio: ' + e.error.message }]);
          setIsRecording(false);
          setIsTranscribing(false);
        };

        recorder.start();
        setMediaRecorder(recorder);
        setIsRecording(true);

        stream.getTracks().forEach((track) => console.log('Track kind:', track.kind, 'state:', track.readyState));
      } catch (error) {
        console.error('Error accessing microphone:', error);
        setMessages((prev) => [...prev, { role: 'assistant', content: 'Error accessing microphone. Please ensure permissions are granted: ' + error.message }]);
        setIsRecording(false);
        setIsTranscribing(false);
      }
    } else {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        if (mediaRecorder.stream) {
          mediaRecorder.stream.getTracks().forEach((track) => {
            if (track.readyState === 'live') {
              track.stop();
              console.log('Stopping track:', track.kind);
            }
          });
        }
      }
      setMediaRecorder(null);
    }
  };

  return (
    <div className="flex flex-col h-full transition-all duration-300">
      <div className="flex-1 overflow-y-auto p-4 bg-gray-800 rounded-lg shadow-inner border border-gray-700">
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
        <AnimatePresence initial={false} mode="sync">
          {messages.map((msg, index) => (
            <motion.div
              key={index}
              className={`mb-6 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <div
                className={`max-w-[75%] p-4 rounded-xl break-words ${
                  msg.role === 'user' ? 'bg-gradient-to-r from-cyan-600 to-blue-700 text-white shadow-lg' : 'bg-gray-700 text-gray-200 shadow-md'
                }`}
                style={{ wordBreak: 'break-word' }}
              >
                {msg.content}
                {msg.audioUrl && msg.role === 'assistant' && (
                  <motion.button
                    onClick={() => handlePlayAudio(msg.audioUrl, index)}
                    className="mt-2 flex items-center text-cyan-400 hover:text-cyan-300 transition duration-200"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    aria-label={playingMessageId === index && isPlaying ? 'Pause Audio Response' : 'Play Audio Response'}
                  >
                    {playingMessageId === index && isPlaying ? (
                      <FaPauseCircle className="mr-2" />
                    ) : (
                      <FaPlayCircle className="mr-2" />
                    )}
                    {playingMessageId === index && isPlaying ? 'Pause Response' : 'Play Response'}
                  </motion.button>
                )}
              </div>
            </motion.div>
          ))}
          {isBotTyping && (
            <motion.div
              key="typing-indicator"
              className="mb-6 flex justify-start"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <div className="max-w-[75%] p-4 rounded-xl bg-gray-700 text-gray-400 italic shadow-md flex items-center">
                <FaSpinner className="animate-spin mr-2" />
                Bot is thinking...
              </div>
            </motion.div>
          )}
          {isTranscribing && (
            <motion.div
              key="transcribing-indicator"
              className="mb-6 flex justify-start"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <div className="max-w-[75%] p-4 rounded-xl bg-gray-700 text-gray-400 italic shadow-md flex items-center">
                <FaSpinner className="animate-spin mr-2" />
                Transcribing audio...
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={chatEndRef} />
      </div>

      <div className="mt-4 flex items-center space-x-3 p-3 bg-gray-800 rounded-lg shadow-lg border border-gray-700">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage(input)}
          placeholder={isRecording ? 'Recording...' : isTranscribing ? 'Processing audio...' : isBotTyping ? 'Waiting for response...' : 'Ask a question...'}
          className={`flex-1 p-3 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 ${
            isRecording ? 'focus:ring-red-500' : 'focus:ring-cyan-500'
          } transition duration-200 placeholder-gray-500`}
          disabled={isRecording || isTranscribing || isSending || isBotTyping}
        />
        <motion.button
          onClick={handleRecord}
          className={`p-3 rounded-full transition duration-200 ease-in-out flex items-center justify-center ${
            isRecording ? 'bg-red-600 hover:bg-red-700 animate-pulse' : 'bg-cyan-600 hover:bg-cyan-700'
          }`}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          aria-label={isRecording ? 'Stop Recording' : 'Start Recording'}
          disabled={isTranscribing || isSending || isBotTyping}
        >
          {isRecording ? <FaStopCircle className="text-white text-xl" /> : <FaMicrophone className="text-white text-xl" />}
        </motion.button>
        <motion.button
          onClick={() => handleSendMessage(input)}
          className={`p-3 bg-cyan-600 rounded-full hover:bg-cyan-700 transition duration-200 ease-in-out flex items-center justify-center ${
            isSending || !input.trim() || isRecording || isTranscribing || isBotTyping ? 'opacity-50 cursor-not-allowed' : ''
          }`}
          whileHover={{ scale: !input.trim() || isSending || isRecording || isTranscribing || isBotTyping ? 1 : 1.1 }}
          whileTap={{ scale: !input.trim() || isSending || isRecording || isTranscribing || isBotTyping ? 1 : 0.9 }}
          disabled={!input.trim() || isSending || isRecording || isTranscribing || isBotTyping}
          aria-label="Send Message"
        >
          <motion.div
            initial={{ rotate: 0 }}
            animate={{ rotate: isSending ? 360 : 0 }}
            transition={{ duration: 0.5, loop: isSending ? Infinity : 0, ease: 'linear' }}
          >
            <FaPaperPlane className="text-white text-xl" />
          </motion.div>
        </motion.button>
      </div>
    </div>
  );
};

export default Chatbot;