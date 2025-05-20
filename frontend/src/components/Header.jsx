import { useState } from 'react';
import axios from 'axios';
import ConfirmationModal from './ConfirmationModal';
import { motion } from 'framer-motion';
import { FaTrashAlt, FaVolumeMute, FaSpinner } from 'react-icons/fa';

const Header = ({ onVoiceModelChange }) => {
  const [isDataModalOpen, setIsDataModalOpen] = useState(false);
  const [isAudioModalOpen, setIsAudioModalOpen] = useState(false);
  const [isDataClearing, setIsDataClearing] = useState(false);
  const [isAudioClearing, setIsAudioClearing] = useState(false);
  const [selectedVoiceModel, setSelectedVoiceModel] = useState('female');

  const voiceModels = [
    { label: 'Female Voice', value: 'female' },
    { label: 'Male Voice', value: 'male' },
  ];

  const handleClearData = async () => {
    setIsDataModalOpen(false);
    setIsDataClearing(true);

    try {
      const response = await axios.post('http://localhost:8000/clear');
      alert(response.data?.message || 'Data cleared successfully.');
    } catch (error) {
      console.error("Error clearing data:", error);
      alert('Error clearing data: ' + (error.response?.data?.detail || error.message || 'An unexpected error occurred.'));
    } finally {
      setIsDataClearing(false);
    }
  };

  const handleClearAudio = async () => {
    setIsAudioModalOpen(false);
    setIsAudioClearing(true);

    try {
      const response = await axios.post('http://localhost:8000/clear_audio');
      alert(response.data?.message || 'Audio files cleared successfully.');
    } catch (error) {
      console.error("Error clearing audio files:", error);
      alert('Error clearing audio files: ' + (error.response?.data?.detail || error.message || 'An unexpected error occurred.'));
    } finally {
      setIsAudioClearing(false);
    }
  };

  const handleVoiceModelChange = (e) => {
    const newVoiceModel = e.target.value;
    console.log('Selected voice model in Header:', newVoiceModel); // Debug log
    setSelectedVoiceModel(newVoiceModel);
    onVoiceModelChange(newVoiceModel);
  };

  return (
    <motion.header
      className="bg-gray-900 p-6 flex justify-between items-center shadow-xl border-b border-gray-700"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <h1 className="text-3xl font-extrabold text-cyan-400 tracking-wide">
        QuerySQL
      </h1>
      <div className="flex space-x-4 items-center">
        <div className="relative">
          <select
            value={selectedVoiceModel}
            onChange={handleVoiceModelChange}
            className="bg-gray-700 text-white font-semibold py-3 px-4 rounded-lg transition duration-200 ease-in-out shadow-md appearance-none focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            {voiceModels.map((model) => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
          <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
            <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
            </svg>
          </div>
        </div>
        <motion.button
          onClick={() => setIsAudioModalOpen(true)}
          className={`bg-yellow-700 hover:bg-yellow-800 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 ease-in-out shadow-md flex items-center ${isAudioClearing ? 'opacity-50 cursor-not-allowed' : ''}`}
          whileHover={{ scale: isAudioClearing ? 1 : 1.05 }}
          whileTap={{ scale: isAudioClearing ? 1 : 0.95 }}
          disabled={isAudioClearing}
          aria-label="Clear Audio Files"
        >
          {isAudioClearing ? (
            <FaSpinner className="animate-spin mr-2" />
          ) : (
            <FaVolumeMute className="mr-2" />
          )}
          {isAudioClearing ? 'Clearing...' : 'Clear Audio'}
        </motion.button>
        <motion.button
          onClick={() => setIsDataModalOpen(true)}
          className={`bg-red-700 hover:bg-red-800 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 ease-in-out shadow-md flex items-center ${isDataClearing ? 'opacity-50 cursor-not-allowed' : ''}`}
          whileHover={{ scale: isDataClearing ? 1 : 1.05 }}
          whileTap={{ scale: isDataClearing ? 1 : 0.95 }}
          disabled={isDataClearing}
          aria-label="Clear All Data"
        >
          {isDataClearing ? (
            <FaSpinner className="animate-spin mr-2" />
          ) : (
            <FaTrashAlt className="mr-2" />
          )}
          {isDataClearing ? 'Clearing...' : 'Clear Data'}
        </motion.button>
      </div>

      <ConfirmationModal
        isOpen={isDataModalOpen}
        onConfirm={handleClearData}
        onCancel={() => setIsDataModalOpen(false)}
        type="data"
      />
      <ConfirmationModal
        isOpen={isAudioModalOpen}
        onConfirm={handleClearAudio}
        onCancel={() => setIsAudioModalOpen(false)}
        type="audio"
      />
    </motion.header>
  );
};

export default Header;