import React, { useEffect, useState } from 'react';
import api from '../api/api';
import SharePopup from './SharePopup';
import JSZip from 'jszip'; 

const FileList = () => {
  const [files, setFiles] = useState([]);
  const [showSharePopup, setShowSharePopup] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSize, setFilterSize] = useState('');
  const [sortField, setSortField] = useState('createdAt');
  const [sortOrder, setSortOrder] = useState('desc');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const filesPerPage = 5;

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const res = await api.get('/files/list');
        setFiles(res.data.files || []);
      }
      catch (err) {
        console.error("Failed to load files", err);
        setFiles([]); 
      }
    };
    fetchFiles();
  }, []);

  const formatDate = (date) => {
    const options = { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return new Date(date).toLocaleDateString('en-US', options);
  };

  const handleDownload = async (file) => {
    try {
      const token = localStorage.getItem('token');
      const res = await api.get(`/files/download/${file._id}`, { responseType: 'blob', headers: { Authorization: `Bearer ${token}` } });
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
  };

  const handleBulkDownload = async () => {
    try {
      const zip = new JSZip();
      selectedFiles.forEach(async (fileId) => {
        const file = files.find(f => f._id === fileId);
        const token = localStorage.getItem('token');
        const res = await api.get(`/files/download/${fileId}`, { responseType: 'blob', headers: { Authorization: `Bearer ${token}` } });
        zip.file(file.fileName, res.data);
      });

      zip.generateAsync({ type: 'blob' }).then(function (content) {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(content);
        link.download = 'files.zip';
        link.click();
      });
    } 
    catch (err) {
      console.error("Bulk download failed", err);
    }
  };

  const handleDelete = async (file) => {
    try {
      await api.delete(`/files/delete/${file._id}`);
      setFiles(prev => prev.filter(f => f._id !== file._id));
    } 
    catch (err) {
      console.error("Delete failed", err);
    }
  };

  const handleBulkDelete = async () => {
    try {
      await Promise.all(selectedFiles.map(fileId => {
        return api.delete(`/files/delete/${fileId}`);
      }));
      setFiles(prev => prev.filter(file => !selectedFiles.includes(file._id)));
      setSelectedFiles([]); 
    } 
    catch (err) {
      console.error("Bulk delete failed", err);
    }
  };

  const handleShare = async (file) => {
    const res = await api.post(`/files/share/${file._id}`);
    const shareUrl = res.data.shareUrl;
    setSelectedFile({ ...file, shareUrl });
    setShowSharePopup(true);
  };

  const closePopup = () => {
    setShowSharePopup(false);
    setSelectedFile(null);
  };

  const handleSelectFile = (fileId) => {
    setSelectedFiles(prev => {
      if (prev.includes(fileId)) {
        return prev.filter(id => id !== fileId);
      } 
      else {
        return [...prev, fileId];
      }
    });
  };

  const filteredFiles = files
    .filter(file => {
      const matchesSearchTerm = file.fileName.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesFileSize = filterSize ? file.size <= parseInt(filterSize) : true;
      const matchesDateRange = (startDate ? new Date(file.createdAt) >= new Date(startDate) : true) && (endDate ? new Date(file.createdAt) <= new Date(endDate) : true);

      return matchesSearchTerm && matchesFileSize && matchesDateRange;
    })
    .sort((a, b) => {
      const aValue = a[sortField];
      const bValue = b[sortField];
      return sortOrder === 'asc' ? (aValue > bValue ? 1 : -1) : (aValue < bValue ? 1 : -1);
    });

  const currentFiles = filteredFiles.slice((currentPage - 1) * filesPerPage, currentPage * filesPerPage);

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  return (
    <div className='bg-gray-50 min-h-screen p-8'>
      <h3 className='text-3xl font-bold text-gray-800 mb-6 text-center'>Your Files</h3>
      <div className='mb-6 flex justify-center flex-wrap'>
        <input type="text" placeholder="Search by file name" value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} className="px-4 py-2 border mr-4 rounded-lg w-1/3 mb-4 sm:mb-0" />
        <select className='px-4 py-2 border rounded-lg mb-4 sm:mb-0' value={filterSize} onChange={(e) => setFilterSize(e.target.value)} >
          <option value=''>All Sizes</option>
          <option value='1000000'>1 MB</option>
          <option value='5000000'>5 MB</option>
          <option value='10000000'>10 MB</option>
        </select>
        <div className='flex space-x-4 justify-center flex-wrap ml-4 mb-4 sm:mb-0'>
          <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="px-4 py-2 mb-4 sm:mb-0 border rounded-lg" />
          <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="px-4 py-2 border rounded-lg" />
        </div>
      </div>
      <div className='mb-4 flex justify-center space-x-4'>
        <button onClick={() => { setSortField('fileName'); setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc'); }} className='px-4 py-2 bg-emerald-500 text-white hover:bg-emerald-600 active:bg-emerald-400 rounded-lg'> Sort by Name</button>
        <button onClick={() => { setSortField('createdAt'); setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc'); }} className='px-4 py-2 bg-emerald-500 text-white hover:bg-emerald-600 active:bg-emerald-400 rounded-lg'>Sort by Date</button>
        <button onClick={() => { setSortField('size'); setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc'); }} className='px-4 py-2 bg-emerald-500 text-white hover:bg-emerald-600 active:bg-emerald-400 rounded-lg'> Sort by Size </button>
      </div>

      {selectedFiles.length > 0 && (
        <div className="mb-4 flex justify-center space-x-4">
          <button onClick={handleBulkDownload} className="px-4 py-2 bg-blue-500 text-white rounded-lg"> Bulk Download </button>
          <button onClick={handleBulkDelete} className="px-4 py-2 bg-red-500 text-white rounded-lg" > Bulk Delete </button>
        </div>
      )}

      {filteredFiles.length === 0 ? (
        <div className='flex justify-center items-center h-full'>
          <p className="text-center text-gray-500">No files match your search or filter criteria.</p>
        </div>
      ) : (
        <div className='overflow-x-auto'>
          <table className='min-w-full bg-white shadow-md rounded-lg'>
            <thead>
              <tr className='bg-gray-100'>
                <th className='py-2 px-4 text-center'>S.No</th>
                <th className='py-2 px-4 text-left'>File Name</th>
                <th className='py-2 px-4 text-center'>Uploaded On</th>
                <th className='py-2 px-4 text-center'>Size</th>
                <th className='py-2 px-4 text-center'>Actions</th>
                <th className='py-2 px-4 text-center'>
                  <input type="checkbox" checked={selectedFiles.length === filteredFiles.length} onChange={() => {
                    if (selectedFiles.length === filteredFiles.length) setSelectedFiles([]);
                    else  setSelectedFiles(filteredFiles.map(file => file._id)) }} />
                </th>
              </tr>
            </thead>
            <tbody>
              {currentFiles.map((file, index) => (
                <tr key={file._id} className="border-b text-center hover:bg-gray-200">
                  <td className="py-2 px-4">{index + 1}</td>
                  <td className="py-2 px-4 text-left">{file.fileName}</td>
                  <td className="py-2 px-4">{formatDate(file.createdAt)}</td>
                  <td className="py-2 px-4">{(file.size / 1000000).toFixed(2)} MB</td>
                  <td className="py-2 px-4">
                    <div className='flex justify-center space-x-4'>
                      <button className='px-4 py-2 bg-gradient-to-r from-blue-400 to-blue-500 text-white rounded-lg hover:from-blue-600 hover:to-blue-400' onClick={() => handleDownload(file)} > Download </button>
                      <button className='px-4 py-2 bg-gradient-to-r from-amber-400 to-amber-500 rounded-lg hover:from-amber-500 hover:to-amber-400' onClick={() => handleShare(file)} > Share </button>
                      <button className='px-4 py-2 bg-gradient-to-r from-red-400 to-red-600 text-white rounded-lg hover:from-red-600 hover:to-red-500' onClick={() => handleDelete(file)} > Delete </button>
                    </div>
                  </td>
                  <td className="py-2 px-4"> <input type="checkbox" checked={selectedFiles.includes(file._id)} onChange={() => handleSelectFile(file._id)} /> </td>
                </tr>
              ))}
      {/*         <div className="sm:hidden"> 
                {currentFiles.map((file) => (
                  <div key={file._id} className="mb-4 p-4 border rounded-md shadow-md">
                    <div className="font-semibold">{file.fileName}</div>
                    <div className="text-gray-500">{formatDate(file.createdAt)}</div>
                    <div className="text-gray-500">{(file.size / 1000000).toFixed(2)} MB</div>
                    <div className="mt-2 flex justify-between">
                      <button onClick={() => handleDownload(file)} className="px-4 py-2 bg-blue-500 text-white rounded-lg">Download</button>
                      <button onClick={() => handleShare(file)} className="px-4 py-2 bg-yellow-500 text-white rounded-lg">Share</button>
                      <button onClick={() => handleDelete(file)} className="px-4 py-2 bg-red-500 text-white rounded-lg">Delete</button>
                    </div>
                  </div>
                ))}
              </div> */}

            </tbody>
          </table>
        </div>
      )}
      <div className="flex justify-center space-x-4 mt-4">
        {Array.from({ length: Math.ceil(filteredFiles.length / filesPerPage) }).map((_, index) => (
          <button key={index} onClick={() => handlePageChange(index + 1)} className={`px-4 py-2 ${currentPage === index + 1 ? 'bg-blue-500' : 'bg-gray-400'} text-white rounded-lg`} > {index + 1} </button>
        ))}
      </div>
      {showSharePopup && selectedFile && (<SharePopup file={selectedFile} onClose={closePopup} /> )}
    </div>
  );
};

export default FileList;