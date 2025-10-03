import { Route, Routes } from "react-router-dom"
import Login from "./pages/Login"
import Register from "./pages/Register"
import PrivateRoute from "./routes/PrivateRoute"
import Dashboard from "./pages/Dashboard"

function App() {

  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/dashboard" element={<PrivateRoute> <Dashboard /> </PrivateRoute>} />
    </Routes>
  )
}

export default App