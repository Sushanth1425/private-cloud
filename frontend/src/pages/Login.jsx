import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import api from '../api/api'
import {signInWithPopup } from 'firebase/auth'
import { auth, googleProvider } from '../firebase'

const Login = () => {
  const [form, setForm]= useState({emailOrUsername:'', password:""})
  const navigate= useNavigate()

  const handleChange= (e)=> setForm({...form, [e.target.name]: e.target.value})

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

  const handleGoolgeLogin= async()=>{
    try {
      const result= await signInWithPopup(auth, googleProvider)
      const token= await result.user.getIdToken()
      const res= await api.post('/auth/google-login', {token})
      localStorage.setItem('token', res.data.token)
      navigate('/dashboard')
    } catch (err) {
      console.error(err)
      alert('Google Login Failed')
    }
  }

  return (
    <div className='flex justify-center items-center min-h-screen bg-gradient-to-r from-indigo-100 to-purple-100'>
      <div className='bg-white p-10 rounded-lg shadow-xl max-w-sm w-full'>
        <h2 className='text-3xl font-extrabold text-gray-900 text-center mb-6'>Login</h2>
        <form onSubmit={handleSubmit}>
          <div className='mb-4'>
            <input className='w-full px-4 py-2 mb-2 border-2 border-gray-300 rounded-lg focus:ring-4 focus:ring-blue-400 focus:outline-none transition duration-300 ease-in-out shadow-md' type="text" name='emailOrUsername' placeholder='Enter Name or Mail' onChange={handleChange} required />
          </div>
          <div className='mb-6'>
            <input className='w-full px-4 py-2 mb-4 border-2 border-gray-300 rounded-lg focus:ring-4 focus:ring-blue-400 focus:outline-none transition duration-300 ease-in-out shadow-md' type="password" name='password' placeholder='Enter password' onChange={handleChange} required />
          </div>
          <button className='w-full py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-semibold rounded-lg hover:from-blue-600 hover:to-indigo-700 transition-all ease-in-out duration-300 shadow-lg transform hover:scale-105' type='submit'>Login</button>
          <button className='w-full py-2 bg-gradient-to-r from-red-500 to-pink-600 text-white font-semibold rounded-lg flex items-center justify-center hover:from-red-600 hover:to-pink-700 transition-all ease-in-out duration-300 shadow-lg transform hover:scale-105 mt-4' type='submit' onClick={handleGoolgeLogin}> <img className='w-7 rounded-xl mr-4 h-7' src="https://e7.pngegg.com/pngimages/337/722/png-clipart-google-search-google-account-google-s-google-play-google-company-text-thumbnail.png" alt="Google Icon" /> Log in with Google</button>
        </form>
        <p className="mt-4 text-center text-gray-600">Don't have an account? <Link className='text-blue-500 hover:text-blue-600' to='/register' >Register</Link></p>
      </div>
    </div>
  )
}

export default Login