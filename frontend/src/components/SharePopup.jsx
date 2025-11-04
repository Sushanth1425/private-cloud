import React, { useState, useEffect } from 'react';

const SharePopup = ({ file, onClose }) => {
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(file.shareUrl)
      .then(() => setIsCopied(true))
      .catch((err) => console.error('Failed to copy text:', err));
  };

  useEffect(() => {
    const handleOutside = e=>{
      if (e.target.classList.contains('bg-gray-500')) onClose();
    }
    window.addEventListener('click', handleOutside);
    return () => window.removeEventListener('click', handleOutside);
  }, [onClose]);

  const formatDate = (date) => {
    if (!date) return '—';
    const options = {year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit'}
    return new Date(date).toLocaleDateString('en-US', options);
  }

  const formatSize = (bytes) => {
    if (!bytes) return '—';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className='fixed inset-0 bg-gray-500 bg-opacity-75 flex justify-center items-center z-50 '>
      <div className='bg-white p-6 rounded-lg shadow-lg w-96'>
        <h3 className='text-2xl font-semibold text-gray-800 mb-4 text-center'> File Shared </h3>
        <p className="mb-2"><strong>File Name:</strong> {file.fileName}</p>
        <p className="mb-2"><strong>File Type:</strong> {file.mimeType}</p>
        <p className="mb-2"><strong>Size:</strong> {formatSize(file.size)}</p>
        {file.shareExpiresAt && (<p className="mb-2"><strong>Expires At:</strong> {formatDate(file.shareExpiresAt)}</p>)}
        <p className="mb-4"><strong>Uploaded At:</strong> {formatDate(file.createdAt)}</p>
        <div className="mb-4">
          <strong>Share Link:</strong>
          <div className="flex space-x-2 mt-1">
            <input type="text" value={file.shareUrl} readOnly className="border px-2 py-1 w-full rounded-lg text-sm" />
            <button onClick={handleCopy} className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition duration-200" > {isCopied ? 'Copied!' : 'Copy'} </button>
          </div>
        </div>
        <button onClick={onClose} className="w-full bg-red-600 text-white py-2 rounded-lg hover:bg-red-500 transition duration-200" > Close </button>
      </div>
    </div>
  );
};

export default SharePopup;