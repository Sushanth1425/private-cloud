const router= require('express').Router()
const bcrypt= require('bcryptjs');
const jwt= require('jsonwebtoken');
const User= require('../models/User');

router.post('/register', async(req, res)=>{
  try{
    const {name, email, password}= req.body
    if (!name || !email || !password) return res.status(400).json({msg:'All fields are required'})
    
    const userExists= await User.findOne({ $or: [{email}, {name}]})
    if (userExists) return res.status(400).json({msg: 'User already exists'})
  
    const salt= await bcrypt.genSalt(10)
    const hashPwd= await bcrypt.hash(password, salt)
  
    const newUser= new User({name, email, password: hashPwd})
    await newUser.save()
  
    const payload= {id: newUser._id}
    const token= jwt.sign(payload, process.env.JWT_SECRET, {expiresIn: '7d'})
    res.json({token})
  }
  catch(err){
    console.error(err)
    return res.status(500).json({msg: 'Server error'})
  }
})

router.post('/login', async(req, res)=>{
  try{
    const {emailOrUsername, password} = req.body
    if (!emailOrUsername || !password) return res.status(400).json({msg: 'All fiends are required!'})
    
    const userExists= await User.findOne({ $or : [{name: emailOrUsername},{email: emailOrUsername}]})
    if (!userExists) return res.status(400).json({msg: 'Invalid Credentials'})

    const validPwd= await bcrypt.compare(password, userExists.password)
    if (!validPwd) return res.status(400).json({msg: 'Invalid Credentials'})

    const payload= {id: userExists._id}
    const token= jwt.sign(payload, process.env.JWT_SECRET, {expiresIn: '7d'})
    res.json({token})
  }
  catch(err){
    console.error(err)
    return res.status(500).json({msg: 'Server error'})
  }
})

module.exports= router