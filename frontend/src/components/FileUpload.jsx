import React, { useState } from 'react'
import api from '../api/api'

const FileUpload = () => {
  const [file, setFile]= useState(null)

  const handleUpload= async()=>{
    const formData= new FormData()
    formData.append('file', file)

    try {
      await api.post('/files/upload', formData, {
        headers: {"Content-Type": "multipart/form-data"}
      })
      alert("File Uploaded Successfully !!")
      window.location.reload()
    } 
    catch (err) {
      console.error(err)
      alert("Upload Failed! Try Again!!")
    }
  }
  return (
    <div className='bg-white p-6 rounded-lg shadow-md w-full max-w-md mx-auto'>
      <h2 className="text-xl font-semibold text-gray-800 mb-4 text-center">Upload a File</h2>
      <input className='w-full px-4 py-2 border border-gray-300 rounded-lg mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500' type="file" onChange={(e)=> setFile(e.target.files[0])} />
      <button className={`w-full py-2 font-semibold rounded-lg transition duration-200 ${file ? 'w-full py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-semibold rounded-lg hover:from-blue-600 hover:to-indigo-700 transition-all ease-in-out duration-300' : 'bg-gray-300 text-gray-500 cursor-not-allowed'}`} onClick={handleUpload} disabled={!file} > Upload </button>
    </div>
  )
}

export default FileUpload;