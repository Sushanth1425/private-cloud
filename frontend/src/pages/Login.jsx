import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import api from '../api/api'

const Login = () => {
  const [form, setForm]= useState({emailOrUsername:'', password:""})
  const navigate= useNavigate()

  const handleChange= (e)=>
    setForm({...form, [e.target.name]: e.target.value})

  const handleSubmit= async(e)=>{
    e.preventDefault();
    try{
      const res= await api.post('/auth/login', form)
      localStorage.setItem("token", res.data.token)
      navigate('/dashboard')
    }
    catch(err){
      console.error(err)
      alert('Login Failed! Try Again!!')
    }
  }

  return (
    <div className='flex justify-center items-center min-h-screen bg-gray-100'>
      <div className='bg-white p-8 rounded-lg shadow-lg w-full max-w-md'>
        <h2 className='text-2xl font-bold text-center text-gray-800 mb-6'>Login</h2>
        <form onSubmit={handleSubmit}>
          <div className='mb-4'>
            <input className='w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500' type="text" name='emailOrUsername' placeholder='Enter Name or Mail' onChange={handleChange} required />
          </div>
          <div className='mb-6'>
            <input className='w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500' type="password" name='password' placeholder='Enter password' onChange={handleChange} required />
          </div>
          <button className='w-full py-2 bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-600 transition duration-200' type='submit'>Login</button>
        </form>
        <p className="mt-4 text-center text-gray-600">Don't have an account? <Link className='text-blue-500 hover:text-blue-600' to='/register' >Register</Link></p>
      </div>
    </div>
  )
}

export default Login