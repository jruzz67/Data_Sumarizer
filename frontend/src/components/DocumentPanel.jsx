import { useState, useRef } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { FaTimes, FaFileUpload, FaSpinner, FaCheckCircle, FaExclamationCircle } from "react-icons/fa";

const supportedTypes = [
  "application/pdf",
  "text/plain",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
];

const DocumentPanel = ({ isOpen, onClose }) => {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [analyzeStatus, setAnalyzeStatus] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isProcessing) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (isProcessing) return;

    const droppedFile = e.dataTransfer.files[0];

    if (droppedFile && supportedTypes.includes(droppedFile.type)) {
      setFile(droppedFile);
      setUploadStatus("");
      setAnalyzeStatus("");
    } else {
      setUploadStatus("Unsupported file type. Please upload PDF, TXT, or Excel.");
      setFile(null);
      setAnalyzeStatus("");
    }
  };

  const handleFileChange = (e) => {
    if (isProcessing) return;

    const selectedFile = e.target.files[0];

    if (selectedFile && supportedTypes.includes(selectedFile.type)) {
      setFile(selectedFile);
      setUploadStatus("");
      setAnalyzeStatus("");
    } else {
      setUploadStatus("Unsupported file type. Please upload PDF, TXT, or Excel.");
      setFile(null);
      setAnalyzeStatus("");
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleUploadAndAnalyze = async () => {
    if (!file || isProcessing) {
      if (!file) setUploadStatus("Please select a file to upload.");
      setAnalyzeStatus("");
      return;
    }

    setIsProcessing(true);
    setUploadStatus("");
    setAnalyzeStatus("");

    const formData = new FormData();
    formData.append("file", file);
    try {
      setUploadStatus("Uploading...");
      const uploadResponse = await axios.post("http://localhost:8000/upload", formData);
      setUploadStatus(uploadResponse.data?.message || "Upload successful.");

      setAnalyzeStatus("Analyzing...");
      const filenameToSend = file.name;
      const analyzeResponse = await axios.post("http://localhost:8000/analyze", {
        filename: filenameToSend,
      });

      if (analyzeResponse.data && typeof analyzeResponse.data.chunk_count === "number") {
        setAnalyzeStatus(`Analysis complete: ${analyzeResponse.data.chunk_count} chunks stored.`);
      } else {
        setAnalyzeStatus("Analysis complete, but chunk count not received.");
      }
    } catch (error) {
      console.error("Error during upload or analysis:", error);
      const errorMessage =
        error.response?.data?.detail || error.message || "An unexpected error occurred.";
      if (uploadStatus.includes("Uploading")) {
        setUploadStatus("Upload failed: " + errorMessage);
        setAnalyzeStatus("");
      } else if (analyzeStatus.includes("Analyzing")) {
        setAnalyzeStatus("Analysis failed: " + errorMessage);
      } else {
        setUploadStatus("Operation failed: " + errorMessage);
        setAnalyzeStatus("");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClearFile = () => {
    if (!isProcessing) {
      setFile(null);
      setUploadStatus("");
      setAnalyzeStatus("");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed top-0 right-0 h-full bg-black text-white shadow-lg z-40 p-6 flex flex-col border-l border-gray-800"
          initial={{ x: "100%" }}
          animate={{ x: 0 }}
          exit={{ x: "100%" }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          style={{ width: "25%", minWidth: "300px" }}
        >
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-white">Document Processing</h2>
            <motion.button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-200 transition duration-200"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              aria-label="Close Panel"
            >
              <FaTimes className="text-2xl" />
            </motion.button>
          </div>

          <div
            className={`border-2 border-dashed rounded-lg p-6 text-center transition duration-200 ease-in-out flex flex-col items-center justify-center ${
              isProcessing
                ? "border-gray-800 bg-gray-900 opacity-70 cursor-not-allowed"
                : isDragging
                ? "border-gray-500 bg-gray-900 cursor-pointer"
                : "border-gray-800 bg-gray-900 hover:border-gray-600 cursor-pointer"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => !isProcessing && fileInputRef.current?.click()}
            role="button"
            aria-label="Select or drop a file"
          >
            <FaFileUpload
              className={`text-5xl mb-4 ${
                isProcessing ? "text-gray-600" : isDragging ? "text-gray-400" : "text-gray-500"
              }`}
            />
            <p className={`mb-2 ${isProcessing ? "text-gray-500" : "text-gray-400"}`}>
              {isProcessing ? "Processing..." : "Drag & Drop your file here"}
            </p>
            {!isProcessing && <p className="text-gray-500 mb-4">or</p>}
            {!isProcessing && (
              <motion.button
                className="bg-gray-700 hover:bg-gray-600 text-white py-2 px-6 rounded-lg transition duration-200 ease-in-out shadow-md"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                aria-label="Browse Files"
                disabled={isProcessing}
              >
                Browse Files
              </motion.button>
            )}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".pdf,.txt,.xlsx,.xls"
              className="hidden"
              disabled={isProcessing}
            />
            {file && (
              <motion.p
                key={file.name}
                className="mt-4 text-gray-300 text-sm truncate w-full"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                Selected: <span className="font-semibold">{file.name}</span>
              </motion.p>
            )}
          </div>

          <div className="mt-4 text-sm flex-grow overflow-y-auto hide-scrollbar">
            <style>
              {`
                .hide-scrollbar::-webkit-scrollbar {
                  display: none;
                }
              `}
            </style>
            <AnimatePresence mode="wait">
              {uploadStatus && (
                <motion.p
                  key="uploadStatus"
                  className={`mt-2 flex items-center ${
                    uploadStatus.includes("failed") ? "text-gray-400" : "text-gray-300"
                  }`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  {uploadStatus.includes("failed") ? (
                    <FaExclamationCircle className="mr-2" />
                  ) : (
                    <FaCheckCircle className="mr-2" />
                  )}
                  {uploadStatus}
                </motion.p>
              )}
              {analyzeStatus && (
                <motion.p
                  key="analyzeStatus"
                  className={`mt-2 flex items-center ${
                    analyzeStatus.includes("failed") ? "text-gray-400" : "text-gray-300"
                  }`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  {analyzeStatus.includes("failed") ? (
                    <FaExclamationCircle className="mr-2" />
                  ) : (
                    <FaCheckCircle className="mr-2" />
                  )}
                  {analyzeStatus}
                </motion.p>
              )}
              {isProcessing && !uploadStatus && !analyzeStatus && (
                <motion.p
                  key="initialProcessing"
                  className="mt-2 flex items-center text-gray-400 italic"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  <FaSpinner className="animate-spin mr-2" />
                  Starting process...
                </motion.p>
              )}
            </AnimatePresence>
          </div>

          <motion.button
            onClick={handleUploadAndAnalyze}
            className={`mt-6 w-full bg-gray-700 hover:bg-gray-600 text-white py-3 rounded-lg transition duration-200 ease-in-out shadow-md flex items-center justify-center ${
              !file || isProcessing ? "opacity-50 cursor-not-allowed" : ""
            }`}
            whileHover={{ scale: !file || isProcessing ? 1 : 1.02 }}
            whileTap={{ scale: !file || isProcessing ? 1 : 0.98 }}
            disabled={!file || isProcessing}
            aria-label="Upload and Analyze"
          >
            {isProcessing ? (
              <FaSpinner className="animate-spin mr-2" />
            ) : (
              <FaFileUpload className="mr-2" />
            )}
            {isProcessing ? "Processing..." : "Upload and Analyze"}
          </motion.button>

          {file && !isProcessing && (
            <motion.button
              onClick={handleClearFile}
              className="mt-4 w-full text-gray-400 hover:text-gray-300 text-sm transition duration-200 ease-in-out"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              aria-label="Clear Selected File"
            >
              Clear Selected File
            </motion.button>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default DocumentPanel;