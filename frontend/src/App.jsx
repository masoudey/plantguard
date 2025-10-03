import React, { useMemo, useState } from "react";
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

  const disableSubmit = useMemo(() => !imageFile || loading, [imageFile, loading]);

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
            Voice Description <span className="font-normal text-slate-500">(optional)</span>
            <input
              type="file"
              accept="audio/wav,audio/mpeg,audio/mp3,audio/x-m4a"
              onChange={(e) => setAudioFile(e.target.files?.[0] ?? null)}
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-sm shadow-inner shadow-slate-100 focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-200"
            />
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
