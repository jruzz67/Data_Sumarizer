import { useState, useEffect, useRef } from "react";
import Header from "./components/Header";
import Chatbot from "./components/Chatbot";
import DocumentPanel from "./components/DocumentPanel";
import { FaChevronRight, FaChevronLeft } from "react-icons/fa";
import { motion } from "framer-motion";
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";

function App() {
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [voiceModel, setVoiceModel] = useState("female");
  const [ws, setWs] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false); // Track connection attempts
  const panelWidth = "min(25vw, 400px)";
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectInterval = 3000; // 3 seconds
  const wsRef = useRef(null); // Use ref to hold WebSocket instance

  const connectWebSocket = () => {
    // Prevent overlapping connection attempts
    if (isConnecting || (wsRef.current && wsRef.current.readyState === WebSocket.OPEN)) {
      console.log("WebSocket connection attempt skipped: already connecting or connected");
      return;
    }

    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      console.error("Max reconnection attempts reached. Please refresh the page.");
      setIsConnected(false);
      setWs(null);
      setIsConnecting(false);
      return;
    }

    setIsConnecting(true);
    console.log("Attempting to connect WebSocket...");
    const websocket = new WebSocket("ws://localhost:8000/ws/voice");
    wsRef.current = websocket;

    websocket.onopen = () => {
      console.log("WebSocket connected");
      setWs(websocket);
      setIsConnected(true);
      setIsConnecting(false);
      reconnectAttemptsRef.current = 0; // Reset attempts on successful connection
    };

    websocket.onclose = () => {
      console.log("WebSocket disconnected");
      setWs(null);
      setIsConnected(false);
      wsRef.current = null;
      setIsConnecting(false);
      reconnectAttemptsRef.current += 1;
      setTimeout(connectWebSocket, reconnectInterval);
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      websocket.close();
      setIsConnecting(false);
    };
  };

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (wsRef.current) {
        if (wsRef.current.readyState === WebSocket.OPEN) {
          console.log("Closing WebSocket on component unmount");
          wsRef.current.send(JSON.stringify({ type: "end" }));
          wsRef.current.close();
        }
        wsRef.current = null;
        setWs(null);
        setIsConnected(false);
        setIsConnecting(false);
      }
    };
  }, []); // Empty dependency array to ensure this runs only on mount/unmount

  const handleVoiceModelChange = (newVoiceModel) => {
    console.log("Voice model updated in App:", newVoiceModel);
    setVoiceModel(newVoiceModel);
    // Chatbot.jsx will handle sending the updated voice model on the next "start"
  };

  const handleDisconnect = () => {
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        console.log("Closing WebSocket via handleDisconnect");
        wsRef.current.send(JSON.stringify({ type: "end" }));
        wsRef.current.close();
      }
      wsRef.current = null;
    }
    setWs(null);
    setIsConnected(false);
    setIsConnecting(false);
    reconnectAttemptsRef.current = 0; // Reset attempts to allow reconnection
    setTimeout(connectWebSocket, reconnectInterval);
  };

  const particlesInit = async (engine) => {
    await loadSlim(engine);
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-black">
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={{
          background: {
            color: {
              value: "#000000",
            },
          },
          fpsLimit: 60,
          particles: {
            color: {
              value: "#ffffff",
            },
            links: {
              color: "#ffffff",
              distance: 150,
              enable: true,
              opacity: 0.5,
              width: 1,
            },
            collisions: {
              enable: true,
            },
            move: {
              direction: "none",
              enable: true,
              outModes: {
                default: "bounce",
              },
              random: false,
              speed: 2,
              straight: false,
            },
            number: {
              density: {
                enable: true,
                area: 800,
              },
              value: 80,
            },
            opacity: {
              value: 0.5,
            },
            shape: {
              type: "circle",
            },
            size: {
              value: { min: 1, max: 5 },
            },
          },
          detectRetina: true,
        }}
        className="absolute inset-0 z-0"
      />

      <div className="relative z-10 flex flex-col min-h-screen text-white">
        <Header
          onVoiceModelChange={handleVoiceModelChange}
          isConnected={isConnected}
        />
        <main className="flex flex-1 overflow-hidden transition-all duration-300 ease-in-out">
          <motion.div
            className="flex-1 p-4 sm:p-6 transition-all duration-300 ease-in-out overflow-y-auto hide-scrollbar"
            initial={{ width: "100%" }}
            animate={{
              width: isPanelOpen ? `calc(100% - ${panelWidth})` : "100%",
            }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
          >
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
            <div className="max-w-full sm:max-w-4xl mx-auto">
              <motion.h2
                className="text-xl sm:text-2xl font-bold text-white mb-4"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                Welcome to QuerySQL
              </motion.h2>
              <motion.p
                className="text-sm sm:text-base text-gray-400 mb-6 sm:mb-8 leading-relaxed"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
                QuerySQL is an intelligent document querying system. Upload your
                documents (PDF, TXT, Excel), and ask questions about their content
                using voice input.
              </motion.p>
              <Chatbot
                isPanelOpen={isPanelOpen}
                voiceModel={voiceModel}
                websocket={ws}
                isConnected={isConnected}
                onDisconnect={handleDisconnect}
              />
            </div>
          </motion.div>
          <motion.button
            onClick={() => setIsPanelOpen(!isPanelOpen)}
            className="fixed top-1/2 transform -translate-y-1/2 bg-gray-700 p-2 sm:p-3 rounded-l-full hover:bg-gray-600 transition duration-300 ease-in-out z-50"
            animate={{ right: isPanelOpen ? panelWidth : "0px" }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            aria-label={
              isPanelOpen ? "Close Document Panel" : "Open Document Panel"
            }
          >
            {isPanelOpen ? (
              <FaChevronRight className="text-white text-lg sm:text-xl" />
            ) : (
              <FaChevronLeft className="text-white text-lg sm:text-xl" />
            )}
          </motion.button>
          <DocumentPanel
            isOpen={isPanelOpen}
            onClose={() => setIsPanelOpen(false)}
          />
        </main>
      </div>
    </div>
  );
}

export default App;