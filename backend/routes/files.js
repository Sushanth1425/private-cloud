const router= require('express').Router()
const multer= require('multer')
const fs= require('fs');
const path= require('path');
const crypto= require('crypto')

const { genDEKHex, generateSM4Key, sm4EncryptHex, wrapDEK, sm4DecryptHex, hashSM3Hex }= require('../utils/crypto');
const { sm2 }= require('sm-crypto')

const File= require('../models/File');
const auth= require('../middlewares/authMiddleware');

const upload= multer({
  dest: 'tmp_uploads/',
  limits: { fileSize: 20*1024*1024}
})

//const sm2Keypair= sm2.generateKeyPairHex()
// for demo only: generate keypair once and persist to disk if not present
const keyFile = path.join(__dirname, '..', 'sm2_keys.json');
let sm2Keypair;
if (fs.existsSync(keyFile)) {
  sm2Keypair = JSON.parse(fs.readFileSync(keyFile, 'utf8'));
} else {
  sm2Keypair = sm2.generateKeyPairHex();
  fs.writeFileSync(keyFile, JSON.stringify(sm2Keypair), 'utf8');
}


// uploading a file
router.post('/upload', auth, upload.single('file'), async(req, res)=>{
  try{
    if (!req.file) return res.status(400).json({msg: 'Select a file to upload'})
    const fileBuffer= fs.readFileSync(req.file.path)

    // 1) generate DEK, encrypt file with SM4
    const dekHex= genDEKHex();
    const cipherHex= sm4EncryptHex(fileBuffer, dekHex);

    // 2) wrap dek with SM2 (using server public key)
    const dekWrapped= wrapDEK(dekHex, sm2Keypair.publicKey)

    // 3) write encrypted blob to storage
    const storageFilename= `${Date.now()}_${req.file.originalname}.enc`
    const storagePath= path.join(process.env.UPLOAD_DIR  || 'uploads', storageFilename)
    fs.writeFileSync(storagePath, Buffer.from(cipherHex, 'hex'))

    // 4) SM3 hash of original 
    const sm3Hash= hashSM3Hex(fileBuffer)

    const fileDoc= await File.create({
      userId: req.user._id,
      fileName: req.file.originalname,
      mimeType: req.file.mimetype,
      storagePath,
      dekWrapped,
      iv: '',
      sm3Hash,
      size: req.file.size
    })

    fs.unlinkSync(req.file.path)
    res.json({msg: 'Upload successful!!', fileId: fileDoc._id})

  }
  catch(err){
    console.error(err)
    return res.status(500).json({ msg: 'Upload error, Try Again!!' });
  }
})

// view uploads
router.get('/list', auth, async(req,res)=>{
  try{
    const files= await File.find({userId: req.user._id}).select('-dekWrapped -__v')
    res.status(200).json({files})
  }
  catch(err){
    console.error(err);
    return res.status(500).json({ msg: 'Failed to list files' });
  }
})

//downlaod a file
router.get('/download/:id', auth, async(req, res)=>{
  try{
    const docExists= await File.findById(req.params.id)
    if (!docExists) return res.status(404).json({msg: 'File not found'})
    if (docExists.userId.toString() !== req.user._id.toString()) return res.status(403).json({msg: 'Forbidden request'})

    // unwrap dek using server private key (in prod use Vault)
    const dekHex= sm2.doDecrypt(docExists.dekWrapped, sm2Keypair.privateKey, 1)
    const cipherHex= fs.readFileSync(docExists.storagePath).toString('hex')
    const plainBuffer= sm4DecryptHex(cipherHex, dekHex)

    const recomputedHash= hashSM3Hex(plainBuffer)
    if (recomputedHash !== docExists.sm3Hash) return res.status(400).json({msg: 'Integrity failed'})

    res.setHeader('Content-disposition', `attachment; filename=${docExists.fileName}"`)
    res.setHeader('Content-Type', docExists.mimeType || 'application/octet-stream')
    return res.end(Buffer.from(plainBuffer));
  }
  catch(err){
    console.error(err)
    res.status(500).json({ msg: 'Failed to download, Try Again!!' });
  }
})

//share a file
router.post('/share/:id', auth, async(req, res)=>{
  try {
    const docExists= await File.findById(req.params.id)
    if (!docExists) return res.status(404).json({msg: 'File not found'})
    if (docExists.userId.toString() !== req.user._id.toString()) return res.status(403).json({msg: 'Forbidden request'})
  
    const token= crypto.randomBytes(18).toString('hex')
    docExists.shareToken= token
    docExists.shareExpiresAt= new Date(Date.now()+ 24*60*60*1000)
    await docExists.save()
  
    const shareUrl= `${req.protocol}://${req.get('host')}/api/files/shared/${token}`
    res.json({shareUrl, expiresAt: docExists.shareExpiresAt})
  }
  catch (err) {
    console.error(err)
    return res.status(500).json({msg: 'Failed to share, Try Again!!'})
  }
})

// shared file
router.get('/shared/:token', async(req, res)=>{
  try{
    const docExists= await File.findOne({shareToken: req.params.token, shareExpiresAt: {$gt: new Date()} })
    if (!docExists) return res.status(404).json({msg: 'Invalid or expired link'})
  
    const dekHex= sm2.doDecrypt(docExists.dekWrapped, sm2Keypair.privateKey, 1)
    const cipherHex= fs.readFileSync(docExists.storagePath).toString('hex')
    const plainBuffer= sm4DecryptHex(cipherHex, dekHex)
  
    const disposition = req.query.download ? "attachment" : "inline";
    res.setHeader('Content-Disposition', `${disposition}; filename=${docExists.fileName}"`)
    res.setHeader('Content-Type', docExists.mimeType || 'application/octet-stream');
    
    return res.end(Buffer.from(plainBuffer));
  }
  catch (err) {
    console.error(err);
    res.status(500).json({ msg: 'Failed to fetch shared file' });
  }

})

// in the files route we used sm2.doDecrypt directly to unwrap — in real deployment you should never store or use the private key in the app; instead, keep it in Vault/Transit and call Vault to unwrap.

// delete a file
router.delete('/delete/:id', auth, async(req, res)=>{
  try{
    const docExists= await File.findById(req.params.id);
    if (!docExists) return res.status(404).json({msg: 'File not found'})
    if (docExists.userId.toString() !== req.user._id.toString()) return res.status(403).json({msg: 'Forbidden Request'})

    if (fs.existsSync(docExists.storagePath)) fs.unlinkSync(docExists.storagePath)

    await docExists.deleteOne()
    
    return res.json({ msg: 'File deleted successfully' });
  }
  catch(err){
    console.error(err)
    return res.status(500).json({msg: 'Failed to delete the file, Try Again!!'})
  }
})

module.exports= router