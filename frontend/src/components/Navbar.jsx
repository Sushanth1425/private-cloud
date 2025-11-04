import { useNavigate } from 'react-router-dom';
import { ReactTyped } from 'react-typed';  

const Navbar = () => {
  const navigate = useNavigate();

  const logout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };

  return (
    <nav className="bg-gradient-to-r from-indigo-600 to-blue-400 p-4 shadow-lg">
      <div className="container mx-auto flex justify-between flex-wrap items-center">
        <h3 className="text-white text-2xl font-bold flex items-baseline">
          <span className="text-red-500 text-4xl mr-1">SuS</span>
          <ReactTyped strings={['- Cloud']} typeSpeed={150} backSpeed={150} backDelay={200} loop />
        </h3>
        <button className="text-white bg-red-500 px-6 font-semibold py-2 rounded-lg hover:bg-red-600 transition-all duration-200 mt-2 sm:mt-0" onClick={logout} > Logout </button>
      </div>
    </nav>
  );
};

export default Navbar;
