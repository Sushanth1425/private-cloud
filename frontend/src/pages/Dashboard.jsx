import React from 'react'
import Navbar from '../components/Navbar'
import FileUpload from '../components/FileUpload'
import FileList from '../components/FileList'

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className='container mx-auto bg-emerald-400 p-12 '>
        <h2 className="text-4xl text-center font-bold text-gray-800 mb-8">Dashboard</h2>
        <div className='space-y-10'>
          <div className="bg-white p-6 rounded-lg shadow-xl transition-transform hover:scale-105 duration-200 ease-in-out"> <FileUpload /> </div>
          <div className="bg-white p-6 rounded-lg shadow-xl "> <FileList /> </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard