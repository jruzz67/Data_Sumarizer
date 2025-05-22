import { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { FaTrashAlt, FaVolumeMute, FaSpinner } from "react-icons/fa";

const Header = ({ onVoiceModelChange, isConnected }) => {
  const [isDataClearing, setIsDataClearing] = useState(false);
  const [isAudioClearing, setIsAudioClearing] = useState(false);
  const [selectedVoiceModel, setSelectedVoiceModel] = useState("female");

  const voiceModels = [
    { label: "Female Voice", value: "female" },
    { label: "Male Voice", value: "male" },
  ];

  const handleClearData = async () => {
    const confirmed = confirm("Are you sure you want to clear all data? This action cannot be undone.");
    if (!confirmed) return;

    setIsDataClearing(true);
    try {
      const response = await axios.post("http://localhost:8000/clear");
      alert(response.data?.message || "Data cleared successfully.");
    } catch (error) {
      console.error("Error clearing data:", error);
      alert(
        "Error clearing data: " +
          (error.response?.data?.detail || error.message || "An unexpected error occurred.")
      );
    } finally {
      setIsDataClearing(false);
    }
  };

  const handleClearAudio = async () => {
    const confirmed = confirm("Are you sure you want to clear all audio files?");
    if (!confirmed) return;

    setIsAudioClearing(true);
    try {
      const response = await axios.post("http://localhost:8000/clear_audio");
      alert(response.data?.message || "Audio files cleared successfully.");
    } catch (error) {
      console.error("Error clearing audio files:", error);
      alert(
        "Error clearing audio files: " +
          (error.response?.data?.detail || error.message || "An unexpected error occurred.")
      );
    } finally {
      setIsAudioClearing(false);
    }
  };

  const handleVoiceModelChange = (e) => {
    const newVoiceModel = e.target.value;
    console.log("Selected voice model in Header:", newVoiceModel);
    setSelectedVoiceModel(newVoiceModel);
    onVoiceModelChange(newVoiceModel);
  };

  return (
    <motion.header
      className="bg-black p-6 flex justify-between items-center shadow-xl border-b border-gray-800"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <h1 className="text-3xl font-extrabold text-white tracking-wide">
        QuerySQL
      </h1>
      <div className="flex space-x-4 items-center">
        <div className="relative">
          <select
            value={selectedVoiceModel}
            onChange={handleVoiceModelChange}
            className="bg-gray-800 text-white font-semibold py-3 px-4 rounded-lg transition duration-200 ease-in-out shadow-md appearance-none focus:outline-none focus:ring-2 focus:ring-gray-500"
          >
            {voiceModels.map((model) => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
          <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
            <svg
              className="w-5 h-5 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M19 9l-7 7-7-7"
              ></path>
            </svg>
          </div>
        </div>
        <div className="text-gray-400">
          Status: {isConnected ? "Connected" : "Disconnected"}
        </div>
        <motion.button
          onClick={handleClearAudio}
          className={`bg-gray-700 hover:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 ease-in-out shadow-md flex items-center ${
            isAudioClearing ? "opacity-50 cursor-not-allowed" : ""
          }`}
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
          {isAudioClearing ? "Clearing..." : "Clear Audio"}
        </motion.button>
        <motion.button
          onClick={handleClearData}
          className={`bg-gray-700 hover:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 ease-in-out shadow-md flex items-center ${
            isDataClearing ? "opacity-50 cursor-not-allowed" : ""
          }`}
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
          {isDataClearing ? "Clearing..." : "Clear Data"}
        </motion.button>
      </div>
    </motion.header>
  );
};

export default Header;