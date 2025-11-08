import React, { useState } from "react";
import api from "../api/api";

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null); // for showing modal result
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setResult(null);

    try {
      await api.post("/files/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult({
        status: "success",
        message: "File Uploaded Successfully ✅",
      });
      setFile(null);
      window.location.reload(); 
    } catch (err) {
      console.error("Upload error:", err);
      const res = err.response;

      if (res && res.data) {
        const data = res.data;

        // --- Malicious file detected ---
        if (res.status === 403) {
          const cls =
            data.prediction_class || data.class || data.label || "Malicious";
          let conf =
            data.confidence ??
            data.malicious_probability ??
            data.confidence_score ??
            data.probability ??
            null;

          if (conf !== null && conf !== undefined) {
            const num = parseFloat(conf);
            if (!isNaN(num)) conf = num <= 1 ? num * 100 : num;
          } else {
            conf = null;
          }

          setResult({
            status: "malicious",
            message: `File detected as ${cls}`,
            confidence: conf ? conf.toFixed(2) : null,
          });
          setLoading(false);
          return;
        }

        // --- Scanner down ---
        if (res.status === 502) {
          setResult({
            status: "error",
            message: "Malware scanner unavailable ⚠️ Try again later.",
          });
          setLoading(false);
          return;
        }

        // --- Generic error ---
        if (data.msg) {
          setResult({
            status: "error",
            message: `Upload failed: ${data.msg}`,
          });
          setLoading(false);
          return;
        }
      }

      // --- Fallback ---
      setResult({
        status: "error",
        message: "Upload Failed! Try Again!!",
      });
    } finally {
      setLoading(false);
    }
  };

  const getRingColor = (confidence) => {
    if (confidence <= 30) return "text-green-500";
    if (confidence <= 70) return "text-yellow-500";
    return "text-red-500";
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-md mx-auto">
      <h2 className="text-xl font-semibold text-gray-800 mb-4 text-center">
        Upload a File
      </h2>

      <input
        className="w-full px-4 py-2 border border-gray-300 rounded-lg mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button
        className={`w-full py-2 font-semibold rounded-lg transition duration-200 ${
          file
            ? "bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:from-blue-600 hover:to-indigo-700"
            : "bg-gray-300 text-gray-500 cursor-not-allowed"
        }`}
        onClick={handleUpload}
        disabled={!file || loading}
      >
        {loading ? "Scanning..." : "Upload"}
      </button>

      {/* Modal / Result */}
      {result && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-2xl shadow-xl w-80 text-center relative">
            <button
              onClick={() => setResult(null)}
              className="absolute top-2 right-3 text-gray-400 hover:text-gray-600 text-lg"
            >
              ✕
            </button>

            {result.status === "malicious" && (
              <>
                <h3 className="text-lg font-semibold text-red-600 mb-4">
                  ⚠️ Malicious File Detected
                </h3>

                {result.confidence ? (
                  <div className="relative w-32 h-32 mx-auto mb-4">
                    <svg
                      className="w-full h-full transform -rotate-90"
                      viewBox="0 0 36 36"
                    >
                      <path
                        className="text-gray-200"
                        strokeWidth="3"
                        stroke="currentColor"
                        fill="none"
                        d="M18 2.0845
                          a 15.9155 15.9155 0 0 1 0 31.831
                          a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                      <path
                        className={getRingColor(result.confidence)}
                        strokeWidth="3"
                        strokeDasharray={`${result.confidence}, 100`}
                        stroke="currentColor"
                        fill="none"
                        d="M18 2.0845
                          a 15.9155 15.9155 0 0 1 0 31.831
                          a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <p className="text-lg font-bold text-gray-700">
                        {result.confidence}%
                      </p>
                      <p className="text-xs text-gray-500">Malicious</p>
                    </div>
                  </div>
                ) : null}

                <p className="text-gray-700 font-medium">
                  {result.message || "Upload blocked!"}
                </p>
              </>
            )}

            {result.status === "success" && (
              <>
                <h3 className="text-lg font-semibold text-green-600 mb-2">
                  ✅ Upload Successful!
                </h3>
                <p className="text-gray-700">{result.message}</p>
              </>
            )}

            {result.status === "error" && (
              <>
                <h3 className="text-lg font-semibold text-yellow-600 mb-2">
                  ⚠️ Upload Error
                </h3>
                <p className="text-gray-700">{result.message}</p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;