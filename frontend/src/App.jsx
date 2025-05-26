import { useState } from 'react';
import Header from './components/Header';
import Chatbot from './components/Chatbot';
import DocumentPanel from './components/DocumentPanel';
import { FaChevronRight, FaChevronLeft } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';
import './BackgroundAnimation.css';

const App = () => {
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [voiceModel, setVoiceModel] = useState('female');
  const panelWidth = 'min(25vw, 400px)';

  const handleVoiceModelChange = (newVoiceModel) => {
    console.log('Voice model updated in App:', newVoiceModel);
    setVoiceModel(newVoiceModel);
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="background-animation-layer">
        <div className="gradient-bg"></div>
        <div className="particles"></div>
      </div>
      <div className="relative z-10 flex flex-col min-h-screen text-gray-200">
        <Header onVoiceModelChange={handleVoiceModelChange} />
        <main className="flex flex-1 overflow-hidden transition-all duration-300 ease-in-out">
          <motion.div
            className="flex-1 p-4 sm:p-6 transition-all duration-300 ease-in-out overflow-y-auto hide-scrollbar"
            initial={{ width: '100%' }}
            animate={{
              width: isPanelOpen ? `calc(100% - ${panelWidth})` : '100%',
            }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
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
                className="text-xl sm:text-2xl font-bold text-cyan-400 mb-4"
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
                using natural language. You can also use voice input to interact
                with the chatbot.
              </motion.p>
              <Chatbot isPanelOpen={isPanelOpen} voiceModel={voiceModel} />
            </div>
          </motion.div>
          <motion.button
            onClick={() => setIsPanelOpen(!isPanelOpen)}
            className="fixed top-1/2 transform -translate-y-1/2 bg-cyan-600 p-2 sm:p-3 rounded-l-full hover:bg-cyan-700 transition duration-300 ease-in-out z-50"
            animate={{ right: isPanelOpen ? panelWidth : '0px' }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            aria-label={
              isPanelOpen ? 'Close Document Panel' : 'Open Document Panel'
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
};

export default App;