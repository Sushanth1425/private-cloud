import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider} from 'firebase/auth'

const firebaseConfig = {
  apiKey: "AIzaSyCCMBgN_X3MjouuxaeSdeoBDYtMnWP-P_o",
  authDomain: "sus-cloud.firebaseapp.com",
  projectId: "sus-cloud",
  storageBucket: "sus-cloud.firebasestorage.app",
  messagingSenderId: "280149699436",
  appId: "1:280149699436:web:b9f468442102c694ce6149",
  measurementId: "G-80Q707BPGX"
};

const app = initializeApp(firebaseConfig);
const auth= getAuth(app)
const googleProvider= new GoogleAuthProvider()

export {auth, googleProvider}