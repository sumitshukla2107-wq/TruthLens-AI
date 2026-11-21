
import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api/v1";

function App() {
  const [file, setFile] = useState(null);
  const [type, setType] = useState("image");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const maxSize = type === "image" ? 10 * 1024 * 1024 : 250 * 1024 * 1024;

  const handleTypeChange = (newType) => {
    setType(newType);
    setFile(null);
    setResult(null);
    setError("");
  };

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0];
    setResult(null);
    setError("");

    if (!selected) {
      setFile(null);
      return;
    }

    const extension = selected.name.split(".").pop().toLowerCase();
    const allowed =
      type === "image"
        ? ["jpg", "jpeg", "png", "webp"]
        : ["mp4", "mov", "avi"];

    if (!allowed.includes(extension)) {
      setFile(null);
      setError(`Unsupported file. Allowed: ${allowed.join(", ")}`);
      event.target.value = "";
      return;
    }

    if (selected.size > maxSize) {
      setFile(null);
      setError(
        type === "image"
          ? "Image must be 10 MB or smaller."
          : "Video must be 250 MB or smaller."
      );
      event.target.value = "";
      return;
    }

    setFile(selected);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append(type === "image" ? "image" : "video", file);

    try {
      const endpoint =
        type === "image" ? "/analyze/image/" : "/analyze/video/";

      const response = await axios.post(API_URL + endpoint, formData);
      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.error ||
          "Upload failed. Check that the Django backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app">
      <header className="navbar">
        <a className="brand" href="/">
          <span className="brand-icon">T</span>
          TruthLens <span>AI</span>
        </a>
        <span className="status">
          <span className="status-dot" /> Backend connected when running
        </span>
      </header>

      <section className="hero">
        <p className="eyebrow">AI MEDIA VERIFICATION</p>
        <h1>
          See beyond the <span>surface.</span>
        </h1>
        <p className="subtitle">
          Upload an image or video to validate its format and inspect basic
          media information.
        </p>
      </section>

      <section className="upload-card">
        <div className="tabs">
          <button
            className={type === "image" ? "tab active" : "tab"}
            onClick={() => handleTypeChange("image")}
          >
            Image analysis
          </button>
          <button
            className={type === "video" ? "tab active" : "tab"}
            onClick={() => handleTypeChange("video")}
          >
            Video analysis
          </button>
        </div>

        <label className="dropzone">
          <span className="upload-icon">{type === "image" ? "▧" : "▶"}</span>
          <strong>{file ? file.name : "Choose a file to get started"}</strong>
          <span>
            {type === "image"
              ? "JPEG, PNG or WebP · Maximum 10 MB"
              : "MP4, MOV or AVI · Maximum 250 MB"}
          </span>
          <span className="browse">Browse files</span>
          <input
            key={type}
            type="file"
            accept={
              type === "image"
                ? ".jpg,.jpeg,.png,.webp"
                : ".mp4,.mov,.avi"
            }
            onChange={handleFileChange}
          />
        </label>

        {file && (
          <div className="file-info">
            <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
            <button className="remove" onClick={() => setFile(null)}>
              Remove
            </button>
          </div>
        )}

        <button
          className="upload-button"
          onClick={handleUpload}
          disabled={!file || loading}
        >
          {loading ? "Uploading and validating..." : "Upload and inspect"}
        </button>

        {error && <div className="message error">{error}</div>}

       
        {result && (
          <div className="result">
            <h2>
  {result.status === "analyzed"
    ? `${type === "video" ? "Video" : "Image"} analysis result`
    : "Upload result"}
</h2>

            <p>
              <strong>Status:</strong> {result.status}
            </p>

            {result.format && (
              <p>
                <strong>Image format:</strong> {result.format}
              </p>
            )}

            {result.size_mb !== undefined && (
              <p>
                <strong>Video size:</strong> {result.size_mb} MB
              </p>
            )}

            {result.frame_count !== undefined && (
              <p>
                <strong>Video frames:</strong> {result.frame_count}
              </p>
            )}

            {result.fps !== undefined && result.fps !== null && (
              <p>
                <strong>Frame rate:</strong> {result.fps} FPS
              </p>
            )}
            {result.frames_analyzed !== undefined && (
  <p>
    <strong>Frames analyzed:</strong>{" "}
    {result.frames_analyzed}
  </p>
)}

            {result.model_status === "ready" &&
            result.prediction ? (
              <div className="prediction">
                <h3>AI detection result</h3>

                <p>
                  <strong>Prediction:</strong>{" "}
                  {result.prediction}
                </p>

                <div className="probability">
                  <div className="probability-label">
                    <span>Real probability</span>
                    <strong>
                      {result.real_probability}%
                    </strong>
                  </div>
                  <div className="probability-track">
                    <div
                      className="probability-fill real-fill"
                      style={{
                        width: `${result.real_probability}%`,
                      }}
                    />
                  </div>
                </div>

                                <div className="probability">
                  <div className="probability-label">
                    <span>AI-generated probability</span>
                    <strong>
                      {(type === "video"
                        ? result.fake_probability
                        : result.ai_probability) ?? 0}%
                    </strong>
                  </div>

                  <div className="probability-track">
                    <div
                      className="probability-fill ai-fill"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(
                            0,
                            Number(
                              type === "video"
                                ? result.fake_probability
                                : result.ai_probability
                            ) || 0
                          )
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <p className="notice">
                  {result.message}
                </p>

                <p className="notice">
                  These are model estimates, not proof of authenticity.
                </p>
              </div>
            ) : (
              <p className="notice">
                {result.message ||
                  "The detection model is not connected for this analysis."}
              </p>
            )}
          </div>
        )}
      </section>

      <footer>
        TruthLens AI · Media validation prototype
      </footer>
    </main>
  );
}

export default App;