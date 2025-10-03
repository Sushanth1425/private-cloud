import React from 'react'
import Navbar from '../components/Navbar'
import FileUpload from '../components/FileUpload'
import FileList from '../components/FileList'

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className='container mx-auto p-4'>
        <h2 className="text-4xl text-center font-semibold text-gray-800 mb-6">Dashboard</h2>
        <div className='space-y-6'>
          <div className="bg-white p-6 rounded-lg shadow-md"> <FileUpload /> </div>
          <div className="bg-white p-6 rounded-lg shadow-md"> <FileList /> </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard