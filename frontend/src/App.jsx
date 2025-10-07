import React, { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import { ResultsPanel } from "./components/ResultsPanel";
import { HistoryPanel } from "./components/HistoryPanel";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export default function App() {
  const [imageFile, setImageFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [history, setHistory] = useState([]);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState(null);
  const [recording, setRecording] = useState(false);
  const [recorderError, setRecorderError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioStreamRef = useRef(null);
  const audioInputRef = useRef(null);
  const ignoreRecorderRef = useRef(false);

  const disableSubmit = useMemo(
    () => !imageFile || loading || recording,
    [imageFile, loading, recording]
  );

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        ignoreRecorderRef.current = true;
        mediaRecorderRef.current.stop();
      }
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach((track) => track.stop());
        audioStreamRef.current = null;
      }
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }
    };
  }, [audioPreviewUrl]);

  const clearAudio = () => {
    setAudioFile(null);
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
      setAudioPreviewUrl(null);
    }
    if (audioInputRef.current) {
      audioInputRef.current.value = "";
    }
  };

  const handleAudioFileChange = (event) => {
    const file = event.target.files?.[0] ?? null;
    clearAudio();
    if (file) {
      setAudioFile(file);
      const url = URL.createObjectURL(file);
      setAudioPreviewUrl(url);
    }
    setRecorderError(null);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      ignoreRecorderRef.current = true;
      mediaRecorderRef.current.stop();
    }
  };

  const handleStartRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setRecorderError("Recording is not supported in this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStreamRef.current = stream;
      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : undefined,
      });
      audioChunksRef.current = [];
      ignoreRecorderRef.current = false;
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        setRecording(false);
        mediaRecorderRef.current = null;
        if (audioStreamRef.current) {
          audioStreamRef.current.getTracks().forEach((track) => track.stop());
          audioStreamRef.current = null;
        }
        const shouldIgnore = ignoreRecorderRef.current;
        ignoreRecorderRef.current = false;
        const chunks = audioChunksRef.current;
        audioChunksRef.current = [];
        if (shouldIgnore || !chunks.length) {
          return;
        }
        const mimeType = recorder.mimeType || "audio/webm";
        const extension = mimeType.includes("mp4")
          ? "m4a"
          : mimeType.includes("mpeg")
          ? "mp3"
          : mimeType.includes("ogg")
          ? "ogg"
          : "webm";
        const blob = new Blob(chunks, { type: mimeType });
        const file = new File([blob], `plantguard-recording-${Date.now()}.${extension}`, {
          type: mimeType,
        });
        clearAudio();
        setAudioFile(file);
        const url = URL.createObjectURL(blob);
        setAudioPreviewUrl(url);
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
      setRecorderError(null);
      clearAudio();
      if (audioInputRef.current) {
        audioInputRef.current.value = "";
      }
    } catch (err) {
      console.error(err);
      setRecorderError("Microphone access was denied.");
    }
  };

  const handleStopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      ignoreRecorderRef.current = false;
      recorder.stop();
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    setLoading(true);

    const form = new FormData();
    if (imageFile) {
      form.append("image", imageFile);
    }
    if (audioFile) {
      form.append("audio", audioFile);
    }
    if (question.trim()) {
      form.append("question", question.trim());
    }

    try {
      const { data } = await axios.post(`${API_BASE}/multimodal/diagnose`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResponse(data);
      setHistory((prev) => [...prev, data]);
    } catch (err) {
      console.error(err);
      setError(
        err.response?.data?.detail ?? "Failed to reach PlantGuard backend. Please retry."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto flex max-w-5xl flex-col gap-8 px-4 pb-16 pt-12 sm:px-6 lg:px-8">
        <header className="flex flex-col items-start gap-4 rounded-2xl bg-white p-8 shadow-lg shadow-slate-200 sm:flex-row sm:items-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald-100">
            <img src="/plantguard.svg" alt="PlantGuard" className="h-12 w-12" />
          </div>
          <div>
            <h1 className="text-3xl font-semibold text-slate-900">PlantGuard Assistant</h1>
            <p className="mt-2 max-w-2xl text-base text-slate-600">
              Upload a symptomatic leaf image, optionally attach a voice description, and ask follow-up
              questions to receive AI-assisted guidance on foliar diseases.
            </p>
          </div>
        </header>

        <form onSubmit={handleSubmit} className="card grid gap-6">
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-semibold text-slate-900">Submit Observations</h2>
            <span className="rounded-full bg-emerald-100 px-3 py-1 text-sm font-medium text-emerald-700">
              Beta Prototype
            </span>
          </div>

          <label className="grid gap-2 text-sm font-medium text-slate-700">
            Leaf Image <span className="font-normal text-slate-500">(required)</span>
            <input
              required
              type="file"
              accept="image/png,image/jpeg"
              onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-sm shadow-inner shadow-slate-100 focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-200"
            />
          </label>

          <label className="grid gap-2 text-sm font-medium text-slate-700">
            Voice Description <span className="font-normal text-slate-500">(optional — record or upload)</span>
            <div className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4 shadow-inner shadow-slate-100">
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={recording ? handleStopRecording : handleStartRecording}
                  className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold shadow transition focus:outline-none focus:ring-2 focus:ring-emerald-200 ${
                    recording
                      ? "bg-rose-600 text-white shadow-rose-300 hover:bg-rose-700"
                      : "bg-emerald-600 text-white shadow-emerald-300 hover:bg-emerald-700"
                  }`}
                >
                  <span className="inline-block h-2 w-2 rounded-full bg-white" />
                  {recording ? "Stop Recording" : "Record Voice"}
                </button>
                <input
                  ref={audioInputRef}
                  type="file"
                  accept="audio/wav,audio/mpeg,audio/mp3,audio/x-m4a,audio/webm"
                  onChange={handleAudioFileChange}
                  disabled={recording}
                  className="w-full flex-1 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm shadow-inner shadow-slate-100 focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-200 disabled:cursor-not-allowed disabled:bg-slate-100 sm:w-auto"
                />
                {audioFile ? (
                  <button
                    type="button"
                    onClick={clearAudio}
                    className="text-sm font-medium text-rose-600 transition hover:text-rose-700"
                  >
                    Clear Audio
                  </button>
                ) : null}
              </div>
              {recording ? (
                <p className="text-sm font-medium text-rose-600">Recording in progress… speak now.</p>
              ) : null}
              {recorderError ? (
                <p className="text-sm text-rose-600">{recorderError}</p>
              ) : null}
              {audioPreviewUrl ? (
                <audio
                  className="w-full"
                  controls
                  src={audioPreviewUrl}
                  aria-label="Recorded audio preview"
                />
              ) : audioFile ? (
                <p className="text-sm text-slate-600">Selected file: {audioFile.name}</p>
              ) : null}
            </div>
          </label>

          <label className="grid gap-2 text-sm font-medium text-slate-700">
            Follow-up Question <span className="font-normal text-slate-500">(optional)</span>
            <textarea
              placeholder="Describe additional concerns or ask about treatment options."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="min-h-[140px] w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-sm shadow-inner shadow-slate-100 focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-200"
            />
          </label>

          <button
            type="submit"
            disabled={disableSubmit}
            className="inline-flex w-full items-center justify-center rounded-full bg-brand px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-200 transition hover:-translate-y-0.5 hover:bg-emerald-700 hover:shadow-emerald-300 disabled:cursor-not-allowed disabled:bg-slate-400 disabled:shadow-none sm:w-max"
          >
            {loading ? "Running analysis…" : "Run Diagnosis"}
          </button>

          {error ? <p className="text-sm text-rose-600">{error}</p> : null}
        </form>

        <ResultsPanel response={response} />
        <HistoryPanel history={history} />

        <footer className="py-6 text-center text-sm text-slate-500">
          Prototype interface for the PlantGuard multimodal chatbot.
        </footer>
      </div>
    </div>
  );
}
