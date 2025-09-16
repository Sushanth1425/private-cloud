const mongoose= require('mongoose')

const connectDB= async()=>{
  try{
    await mongoose.connect(process.env.MONGO_URI)
    console.log(`DB connected to ${mongoose.connection.host}`)
  }
  catch(err){
    console.error(err)
  }
}

module.exports= connectDB