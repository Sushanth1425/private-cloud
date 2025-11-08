import React, { useState, useEffect } from 'react'
import api from '../api/api'
import FileUpload from './FileUpload'
import FileList from './FileList'

const FilesPage = () => {
  const [files, setFiles] = useState([])

  const fetchFiles = async () => {
    try {
      const res = await api.get('/files/list')
      setFiles(res.data.files || [])
    } 
    catch (err) {
      console.error("Failed to load files", err)
      setFiles([])
    }
  }

  useEffect(() => {
    fetchFiles()
  }, [])

  const handleUploadSuccess = () => {
    fetchFiles() 
  }

  return (
    <div className="min-h-screen bg-gray-100 py-10">
      <div className="max-w-5xl mx-auto space-y-8">
        <FileUpload onUploadSuccess={handleUploadSuccess} />
        <FileList files={files} setFiles={setFiles} />
      </div>
    </div>
  )
}

export default FilesPage