import { motion, AnimatePresence, transform } from 'framer-motion';
import { FaCheckCircle, FaTimesCircle } from 'react-icons/fa';

const ConfirmationModal = ({ isOpen, onConfirm, onCancel }) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50 p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          onClick={onCancel} // Enable clicking overlay to cancel
          role="dialog" // Accessibility: Indicate this is a dialog
          aria-modal="true" // Accessibility: Indicate modal nature
          aria-labelledby="modal-title" // Reference the title for screen readers
        >
          <motion.div
            className="bg-gray-900 rounded-xl shadow-2xl p-8 w-full max-w-sm border border-gray-700"
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 50, opacity: 0 }}
            transition={{ duration: 0.3 }}
            onClick={(e) => e.stopPropagation()} // Prevent overlay click from closing when clicking modal
          >
            <h2 id="modal-title" className="text-2xl font-bold text-cyan-400 mb-6 text-center">
              Confirm Action
            </h2>
            <p className="text-gray-300 text-center mb-8 leading-relaxed">
              Are you sure you want to clear all data? This action cannot be undone.
            </p>
            <div className="flex justify-center space-x-6">
              <motion.button
                onClick={onConfirm}
                className="flex items-center px-6 py-3 bg-green-700 text-white rounded-lg shadow-md hover:bg-green-800 transition duration-200 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                aria-label="Confirm"
              >
                <FaCheckCircle className="mr-2 text-xl" />
                Confirm
              </motion.button>
              <motion.button
                onClick={onCancel}
                className="flex items-center px-6 py-3 bg-red-700 text-white rounded-lg shadow-md hover:bg-red-800 transition duration-200 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                aria-label="Cancel"
              >
                <FaTimesCircle className="mr-2 text-xl" />
                Cancel
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default ConfirmationModal;