import React from "react";

const formatConfidence = (value) => `${(value * 100).toFixed(1)}%`;

export function ResultsPanel({ response }) {
  if (!response) {
    return null;
  }

  return (
    <div className="card space-y-6">
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold text-slate-900">Fusion Diagnosis</h3>
          <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium uppercase tracking-wide text-emerald-700">
            Combined
          </span>
        </div>
        {response.fusion?.top_k?.length ? (
          <ul className="space-y-2">
            {response.fusion.top_k.map((item) => (
              <li
                key={item.label}
                className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 px-4 py-3 text-sm"
              >
                <span className="font-medium text-slate-800">{item.label}</span>
                <span className="inline-flex items-center rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-700">
                  {formatConfidence(item.confidence)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-500">No fusion prediction available.</p>
        )}
      </section>

      {response.vision?.top_k?.length ? (
        <section className="space-y-3">
          <h3 className="text-lg font-semibold text-slate-900">Vision Model</h3>
          <ul className="space-y-2">
            {response.vision.top_k.map((item) => (
              <li
                key={item.label}
                className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 px-4 py-3 text-sm"
              >
                <span className="font-medium text-slate-800">{item.label}</span>
                <span className="inline-flex items-center rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-700">
                  {formatConfidence(item.confidence)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {response.transcript ? (
        <section className="space-y-2">
          <h3 className="text-lg font-semibold text-slate-900">Transcribed Symptoms</h3>
          <p className="rounded-lg border border-slate-100 bg-slate-50 p-4 text-sm text-slate-700">
            {response.transcript}
          </p>
        </section>
      ) : null}

      {response.source ? (
        <section className="space-y-2">
          <h3 className="text-lg font-semibold text-slate-900">Primary Evidence</h3>
          <p className="text-sm text-slate-600">
            Source: <span className="font-medium text-slate-800">{response.source.source || 'retriever'}</span>
          </p>
          <p className="text-sm text-slate-700 whitespace-pre-wrap">{response.context}</p>
        </section>
      ) : null}

      {response.sources?.length ? (
        <section className="space-y-3">
          <h3 className="text-lg font-semibold text-slate-900">Knowledge Sources</h3>
          <ul className="space-y-2">
            {response.sources.map((item) => (
              <li
                key={`${item.source}-${item.chunk_id}`}
                className="rounded-lg border border-slate-200 bg-white p-3 text-sm shadow-sm"
              >
                <div className="flex items-center justify-between text-xs text-slate-500">
                  <span>{item.source || 'retriever'}</span>
                  {item.retrieval_score !== undefined ? (
                    <span>score: {item.retrieval_score.toFixed?.(3) ?? item.retrieval_score}</span>
                  ) : null}
                </div>
                <p className="mt-2 whitespace-pre-wrap text-slate-700 max-h-32 overflow-hidden">{item.text}</p>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
