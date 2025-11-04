const router= require('express').Router()
const bcrypt= require('bcryptjs');
const jwt= require('jsonwebtoken');
const User= require('../models/User');
const admin= require('../firebaseAdmin')

router.post('/google-login', async(req, res)=>{
  try {
    const {token}= req.body;
    const decodedToken= await admin.auth().verifyIdToken(token)
    let user= await User.findOne({email: decodedToken.email})

    if (!user){
      user= new User({
        name: decodedToken.name,
        email: decodedToken.email,
        password: '',
      })
      await user.save();
    }
    const payload= {id: user._id}
    const jwtToken= jwt.sign(payload, process.env.JWT_SECRET, {expiresIn: '7d'})

    res.json({ token: jwtToken });
  } 
  catch (err) {
    console.error(err.message);
    return res.status(500).json({ msg: 'Google Login Failed' })
  }
})

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