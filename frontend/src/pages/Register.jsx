import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import api from '../api/api'

const Register = () => {
  const [form, setForm]= useState({name:'', email:'', password:''})

  const navigate= useNavigate()

  const handleSubmit= async(e)=>{
    e.preventDefault();
    try {
      await api.post('/auth/register', form)
      navigate('/')
    } 
    catch (err) {
      console.error(err)
      alert('Registration Failed! Try Again!!')
    }
  }

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-100">
      <div className='bg-white p-8 rounded-lg shadow-lg w-full max-w-md'>
        <h2 className='text-2xl font-bold text-center text-gray-800 mb-6'>Register</h2>
        <form onSubmit={handleSubmit}>
          <div className='mb-4'> <input type="text" name="name" placeholder='Enter Name' className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" onChange={(e)=>setForm({...form, [e.target.name]: e.target.value})} /> </div>
          <div className="mb-4"> <input type="email" name="email" placeholder='Enter email' className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" onChange={(e)=>setForm({...form, [e.target.name]: e.target.value})} /> </div>
          <div className="mb-6"> <input type="password" name="password" placeholder='Enter password' className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" onChange={(e)=>setForm({...form, [e.target.name]: e.target.value})} /> </div>
          <button type='submit' className='w-full py-2 bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-600 transition duration-200'>Register</button>
        </form>
        <p className='mt-4 text-center text-gray-600'>Already Registered? <Link className='text-blue-500 hover:text-blue-600' to='/'>Login</Link></p>
      </div>
    </div>
  )
}

export default Register