import React, { useEffect, useState } from 'react';
import api from '../api/api';
import SharePopup from './SharePopup';

const FileList = () => {
  const [files, setFiles] = useState([]);
  const [showSharePopup, setShowSharePopup] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const res = await api.get('/files/list');
        setFiles(res.data.files);
      } 
      catch (err) {
        console.error("Failed to load files", err);
      }
    }
    fetchFiles();
  }, []);

  const formatDate = (date)=> {
    const options = {year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit'}
    return new Date(date).toLocaleDateString('en-US', options);
  }

  const handleDownload = async (file) => {
    try {
      const token = localStorage.getItem('token');
      const res = await api.get(`/files/download/${file._id}`, {responseType: 'blob', headers: { Authorization: `Bearer ${token}`}})
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute("download", file.fileName);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } 
    catch (err) {
      console.error("Download failed", err);
    }
  }

  const handleDelete = async (file) => {
    try {
      await api.delete(`/files/delete/${file._id}`);
      setFiles(prev => prev.filter(f => f._id !== file._id));
    } 
    catch (err) {
      console.error("Delete failed", err);
    }
  }

  const handleShare = async (file) => {
    const res = await api.post(`/files/share/${file._id}`);
    const shareUrl = res.data.shareUrl;
    setSelectedFile({...file, shareUrl, fileType: file.fileType, size: file.size, expiresAt: file.expiresAt, uploadedAt: file.createdAt});
    setShowSharePopup(true);
  }

  const closePopup = () => {
    setShowSharePopup(false)
    setSelectedFile(null)
  }

  return (
    <div className='bg-gray-50 min-h-screen p-4'>
      <h3 className='text-3xl font-semibold text-gray-800 mb-6 text-center'> Your Files </h3>
      {files.length === 0 ? <div className='flex justify-center items-center h-full'> <p className="text-center text-gray-500"> No files uploaded yet. Please upload files to see them here. </p> </div>
       : (
        <div className='overflow-x-auto'>
          <table className='min-w-full bg-white shadow-md rounded-lg'>
            <thead>
              <tr className='bg-gray-100'>
                <th className='py-2 px-4 text-left'>S. No.</th>
                <th className='py-2 px-4 text-left'>File Name</th>
                <th className='py-2 px-4 text-center'>Uploaded On</th>
                <th className='py-2 px-4 text-center'>Actions</th>
              </tr>
            </thead>
            <tbody>
              {files.map((file, index) => (
                <tr key={file._id} className="border-b text-center hover:bg-gray-50">
                  <td className="py-2 px-4 text-left">{index + 1}</td>
                  <td className="py-2 px-4 text-left">{file.fileName}</td>
                  <td className="py-2 px-4">{formatDate(file.createdAt)}</td>
                  <td className="py-2 px-4">
                    <div className='flex align-middle justify-center space-x-4'>
                      <button className='px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition duration-200' onClick={() => handleDownload(file)} > Download </button>
                      <button className='px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition duration-200' onClick={() => handleShare(file)} > Share </button>
                      <button className='px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition duration-200' onClick={() => handleDelete(file)} > Delete </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {showSharePopup && selectedFile && ( <SharePopup file={selectedFile} onClose={closePopup} /> )}
    </div>
  );
};

export default FileList;