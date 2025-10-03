import React from "react";

export function HistoryPanel({ history }) {
  if (!history.length) {
    return null;
  }

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">Conversation History</h2>
        <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
          {history.length} turn{history.length === 1 ? "" : "s"}
        </span>
      </div>
      <ul className="space-y-3">
        {history.map((entry, idx) => (
          <li key={`history-${idx}`} className="rounded-xl border border-slate-100 bg-slate-50 p-4">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Turn {idx + 1}
            </span>
            <pre className="mt-2 whitespace-pre-wrap rounded-lg bg-white p-3 text-xs text-slate-700 shadow-inner shadow-slate-100">
              {JSON.stringify(entry, null, 2)}
            </pre>
          </li>
        ))}
      </ul>
    </div>
  );
}
