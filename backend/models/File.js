const mongoose= require('mongoose')

const FileSchema= mongoose.Schema({
  userId: {type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true},
  fileName: {type: String, required: true},
  mimeType: {type: String, required: true},
  storagePath: {type: String, required: true},
  dekWrapped: {type: String, required: true},
  iv: {type: String},
  sm3Hash: {type: String},
  shareToken: {type: String},
  shareExpiresAt: {type: Date}
}, { timestamps: true })

module.exports= mongoose.model('File', FileSchema)