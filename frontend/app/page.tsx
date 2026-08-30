"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { uploadFiles } from "@/lib/api";
import { Upload, FileText, AlertCircle, CheckCircle2 } from "lucide-react";

interface FileDropState {
  file: File | null;
  dragging: boolean;
}

function FileDrop({
  label,
  id,
  value,
  onChange,
}: {
  label: string;
  id: string;
  value: File | null;
  onChange: (f: File) => void;
}) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file && file.type === "application/pdf") onChange(file);
    },
    [onChange]
  );

  return (
    <div
      id={id}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`relative flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-10 transition-all cursor-pointer
        ${dragging ? "border-violet-400 bg-violet-950/30" : "border-slate-600 bg-slate-800/40"}
        ${value ? "border-emerald-500 bg-emerald-950/20" : ""}
        hover:border-violet-500 hover:bg-violet-950/20`}
      onClick={() => document.getElementById(`${id}-input`)?.click()}
    >
      <input
        id={`${id}-input`}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onChange(f);
        }}
      />
      {value ? (
        <>
          <CheckCircle2 className="w-10 h-10 text-emerald-400" />
          <p className="text-sm font-medium text-emerald-300">{value.name}</p>
          <p className="text-xs text-slate-400">{(value.size / 1024).toFixed(1)} KB</p>
        </>
      ) : (
        <>
          <FileText className="w-10 h-10 text-slate-400" />
          <p className="text-sm font-semibold text-slate-200">{label}</p>
          <p className="text-xs text-slate-500">Drag &amp; drop or click to browse</p>
          <span className="mt-1 rounded-full bg-slate-700 px-3 py-1 text-xs text-slate-300">PDF only</span>
        </>
      )}
    </div>
  );
}

export default function UploadPage() {
  const router = useRouter();
  const [qp, setQp] = useState<File | null>(null);
  const [ans, setAns] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!qp || !ans) return;
    setUploading(true);
    setError(null);
    try {
      const jobId = await uploadFiles(qp, ans);
      router.push(`/evaluate/${jobId}`);
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Upload failed. Is the backend running?");
      setUploading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-violet-950 flex flex-col items-center justify-center px-4">
      {/* Header */}
      <div className="mb-12 text-center">
        <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-violet-900/40 px-4 py-1.5 text-xs font-semibold tracking-widest text-violet-300 uppercase border border-violet-700/50">
          AI-Powered Assessment
        </div>
        <h1 className="text-5xl font-black text-white tracking-tight mb-3">
          Veda<span className="text-violet-400">AI</span>
        </h1>
        <p className="text-slate-400 max-w-md text-sm leading-relaxed">
          Upload a question paper and a student answer sheet. Our AI pipeline will segment, map, and grade every answer automatically.
        </p>
      </div>

      {/* Upload card */}
      <div className="w-full max-w-2xl rounded-3xl bg-slate-900/70 border border-slate-700/50 backdrop-blur-xl shadow-2xl p-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <FileDrop id="qp" label="Question Paper" value={qp} onChange={setQp} />
          <FileDrop id="ans" label="Answer Sheet" value={ans} onChange={setAns} />
        </div>

        {error && (
          <div className="mt-5 flex items-center gap-2 rounded-xl bg-red-950/40 border border-red-700/40 px-4 py-3 text-sm text-red-300">
            <AlertCircle className="w-4 h-4 shrink-0" />
            {error}
          </div>
        )}

        <button
          id="submit-upload"
          onClick={handleSubmit}
          disabled={!qp || !ans || uploading}
          className="mt-6 w-full flex items-center justify-center gap-2 rounded-2xl bg-violet-600 hover:bg-violet-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold py-4 text-sm transition-all duration-200 shadow-lg shadow-violet-900/50"
        >
          {uploading ? (
            <>
              <span className="inline-block w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
              Uploading…
            </>
          ) : (
            <>
              <Upload className="w-4 h-4" />
              Start AI Analysis
            </>
          )}
        </button>
      </div>

      <p className="mt-8 text-xs text-slate-600">
        Processing typically takes 2–5 minutes depending on PDF length.
      </p>
    </main>
  );
}
