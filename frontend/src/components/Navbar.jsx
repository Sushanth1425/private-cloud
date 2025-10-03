import { useNavigate } from 'react-router-dom'

const Navbar = () => {
  const navigate= useNavigate()

  const logout= ()=>{
    localStorage.removeItem('token')
    navigate('/')
  }
  return (
    <nav className='bg-blue-300 p-4 shadow-md'>
      <div className='container mx-auto flex justify-between flex-wrap items-center'>
        <h3 className='text-white text-2xl font-semibold'>SuS-Cloud</h3>
        <button className='text-white bg-red-500 px-4 py-2 rounded-lg hover:bg-red-600 transition duration-200 mt-2 sm:mt-0' onClick={logout}>Logout</button>
      </div>
    </nav>
  )
}

export default Navbar