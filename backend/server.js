const express = require('express')
require('dotenv').config()
const cors= require('cors')
const fs= require('fs')
const helmet= require('helmet')
const rateLimit= require('express-rate-limit')
const morgan= require('morgan')

const connectDB= require('./utils/db')
const authRoutes= require('./routes/auth')
const filesRoutes= require('./routes/files')

const app= express()
connectDB()


app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  methods: ['GET', 'PUT', 'DELETE', 'POST'],
  credentials: true
}))
app.use(helmet())
app.use(morgan('dev'))

const limit= rateLimit({
  windowMs: 10*60*1000,
  max: 100
})
app.use(limit)

app.use(express.json())
app.use(express.urlencoded({ extended: true }))

const uplaod_dir= process.env.UPLOAD_DIR || './uploads'
if (!fs.existsSync(uplaod_dir)) fs.mkdirSync(uplaod_dir, {recursive: true})

app.use('/api/auth',authRoutes)
app.use('/api/files', filesRoutes)

const port= process.env.PORT || 5050
app.listen(port, ()=>console.log(`server connected to ${port}`))