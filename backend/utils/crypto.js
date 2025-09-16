const crypto= require('crypto')
const {sm2, sm3, sm4}= require('sm-crypto')
const { buffer } = require('stream/consumers')

// Generate a random 32-byte DEK (hex)
const genDEKHex= () => crypto.randomBytes(16).toString('hex')

// Encrypt buffer data with SM4 (ECB/CBC/GCM availability depends on lib).
// sm-crypto's sm4.encrypt expects 'plaintext' and 'key' strings and mode.
// We'll use sm4.encrypt for a basic approach; for production prefer AEAD mode (GCM).

/* const sm4EncryptHex=(ptBuffer, dekHex) =>{
  const plainHex= ptBuffer.toString('hex')
  const cipherHex= sm4.encrypt(plainHex, dekHex)
  return cipherHex
} */

// Encrypt buffer data with SM4
const sm4EncryptHex = (ptBuffer, dekHex) => {
  const plainHex = ptBuffer.toString("hex");   // convert buffer → hex
  return sm4.encrypt(plainHex, dekHex);        // returns hex string
};

// Decrypt with SM4
const sm4DecryptHex= (cipherHex, dekHex) =>{
  const plainHex= sm4.decrypt(cipherHex, dekHex)
  return Buffer.from(plainHex, 'hex')
}

// Wrap/unwrap DEK using SM2 (ECIES-like) - uses hex strings
const wrapDEK= (dekHex, sm2PublicKeyHex)=> sm2.doEncrypt(dekHex, sm2PublicKeyHex, 1)

const unwrapDEK= (wrappedDek, sm2PrivateKeyHex)=> sm2.doDecrypt(wrappedDek, sm2PrivateKeyHex, 1)

const hashSM3Hex= (buffer)=> sm3(buffer.toString('hex'))

module.exports= {genDEKHex,
  genDEKHex,
  sm4EncryptHex,
  sm4DecryptHex,
  wrapDEK,
  unwrapDEK,
  hashSM3Hex
}