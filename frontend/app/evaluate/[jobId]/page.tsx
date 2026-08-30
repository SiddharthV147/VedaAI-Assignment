"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import {
  getStatus,
  getResults,
  getPdfUrl,
  flattenQuestions,
  lookupCoords,
  FlatQuestion,
  CoordEntry,
  ResultsPayload,
} from "@/lib/api";
import { Document, Page, pdfjs } from "react-pdf";
import { Loader2, ChevronLeft, ChevronRight, Search } from "lucide-react";

pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.mjs`;

// ─────────────────────────────────────────────────────────────────────────────
// Bounding box canvas overlay
// ─────────────────────────────────────────────────────────────────────────────
function BBoxOverlay({
  coords,
  currentPage,
  ocrW,
  ocrH,
}: {
  coords: CoordEntry[];
  currentPage: number;
  ocrW: number;
  ocrH: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    const canvas = canvasRef.current;
    if (!el || !canvas || !ocrW || !ocrH) return;

    // Observe container size changes to resize canvas and redraw boxes
    const ro = new ResizeObserver(() => {
      const { clientWidth, clientHeight } = el;
      if (!clientWidth || !clientHeight) return;
      canvas.width = clientWidth;
      canvas.height = clientHeight;
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, clientWidth, clientHeight);

      const sx = clientWidth / ocrW;
      const sy = clientHeight / ocrH;

      coords
        .filter((c) => c.page === currentPage)
        .forEach(({ bbox: [x1, y1, x2, y2] }) => {
          const rx = x1 * sx, ry = y1 * sy;
          const rw = (x2 - x1) * sx, rh = (y2 - y1) * sy;
          ctx.fillStyle   = "rgba(139,92,246,0.18)";
          ctx.fillRect(rx, ry, rw, rh);
          ctx.strokeStyle = "rgba(139,92,246,0.9)";
          ctx.lineWidth   = 2.5;
          ctx.strokeRect(rx, ry, rw, rh);
        });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [coords, currentPage, ocrW, ocrH]);

  return (
    <div ref={containerRef} className="absolute inset-0 pointer-events-none">
      <canvas ref={canvasRef} className="w-full h-full block" />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PDF Viewer
// ─────────────────────────────────────────────────────────────────────────────
function PdfViewer({
  jobId,
  coords,
  targetPage,
  pageDims,
}: {
  jobId:     string;
  coords:    CoordEntry[];
  targetPage: number | null;
  pageDims:  Record<string, { width_px: number; height_px: number }>;
}) {
  const [numPages, setNumPages] = useState(0);
  const [renderW, setRenderW] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({});

  // Jump to the answer page when the active question changes
  useEffect(() => {
    if (targetPage && pageRefs.current[targetPage]) {
      pageRefs.current[targetPage]?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [targetPage]);

  // Measure container width to scale PDF
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setRenderW(el.clientWidth - 40)); // padding
    ro.observe(el);
    setRenderW(el.clientWidth - 40);
    return () => ro.disconnect();
  }, []);

  const hasHL = coords.length > 0;

  return (
    <div className="flex flex-col h-full rounded-2xl border border-slate-700/50 bg-slate-950 overflow-hidden">
      {/* toolbar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 shrink-0 gap-3">
        <span className="text-sm font-semibold text-slate-300">Answer Sheet</span>
        
        {hasHL && (
          <div className="flex items-center gap-3">
            <span className="rounded-full bg-violet-700/40 border border-violet-600/40 text-violet-300 text-xs px-3 py-1">
              Answer highlighted
            </span>
            <span className="text-xs text-slate-500 hidden sm:block font-mono">
              Pages: {[...new Set(coords.map(c => c.page))].join(", ")}
            </span>
          </div>
        )}
      </div>

      {/* canvas */}
      <div ref={containerRef} className="flex-1 overflow-y-auto flex flex-col items-center bg-slate-900/60 p-5 scroll-smooth">
        <Document
          file={getPdfUrl(jobId)}
          onLoadSuccess={({ numPages }) => setNumPages(numPages)}
          loading={
            <div className="flex items-center gap-2 text-slate-500 text-sm p-8 mt-10">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading PDF…
            </div>
          }
          className="flex flex-col gap-10 max-w-full pb-10"
        >
          {Array.from({ length: numPages }, (_, i) => i + 1).map((page) => {
            const ocrDim = pageDims[String(page)];
            const isTarget = page === targetPage;
            
            return (
              <div 
                key={page}
                ref={(el) => { pageRefs.current[page] = el; }}
                className={`relative shadow-2xl transition-all duration-300 bg-white ${
                  isTarget ? "ring-4 ring-violet-500 ring-offset-4 ring-offset-slate-900 scale-[1.01]" : ""
                }`}
              >
                <Page
                  pageNumber={page}
                  width={renderW > 0 ? renderW : undefined}
                  renderAnnotationLayer={false}
                  renderTextLayer={false}
                />
                
                {ocrDim && (
                  <BBoxOverlay
                    coords={coords}
                    currentPage={page}
                    ocrW={ocrDim.width_px}
                    ocrH={ocrDim.height_px}
                  />
                )}
                
                {/* Page number badge */}
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 rounded-full bg-slate-800/80 px-3 py-1 text-xs text-slate-400 font-mono">
                  Page {page} / {numPages}
                </div>
              </div>
            );
          })}
        </Document>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Question card
// ─────────────────────────────────────────────────────────────────────────────
function QuestionCard({
  q,
  isActive,
  coords,
  onClick,
}: {
  q: FlatQuestion;
  isActive: boolean;
  coords: CoordEntry[];
  onClick: () => void;
}) {
  const firstPage = coords[0]?.page;

  return (
    <div
      id={`card-${q.path.replace(/\W+/g, "-")}`}
      onClick={onClick}
      style={{ marginLeft: q.depth === 2 ? "1rem" : 0 }}
      className={`rounded-xl border p-3 cursor-pointer transition-all duration-150 ${
        isActive
          ? "border-violet-500 bg-violet-950/35 shadow-md shadow-violet-900/20"
          : "border-slate-700/50 bg-slate-800/30 hover:border-slate-600 hover:bg-slate-800/60"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5 flex-wrap">
            {q.depth === 1 && (
              <span className="text-[10px] font-bold text-violet-400 uppercase tracking-wider">
                {q.section}
              </span>
            )}
            <span className={`font-semibold ${q.depth === 2 ? "text-xs text-slate-400" : "text-xs text-slate-200"}`}>
              {q.label}
            </span>
          </div>
          {q.text && (
            <p className="text-xs text-slate-300 line-clamp-2 leading-relaxed">
              {q.text}
            </p>
          )}
        </div>

        {firstPage && (
          <span className="shrink-0 text-[10px] text-slate-500 mt-0.5">p.{firstPage}</span>
        )}
      </div>

      {coords.length === 0 && (
        <p className="mt-1 text-[10px] text-amber-600">No answer region found</p>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────────────────────
export default function EvaluatePage() {
  const { jobId } = useParams() as { jobId: string };
  const [status,      setStatus]      = useState("pending");
  const [progress,    setProgress]    = useState("Initialising…");
  const [data,        setData]        = useState<ResultsPayload | null>(null);
  const [error,       setError]       = useState<string | null>(null);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [search,      setSearch]      = useState("");

  // Poll backend
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const s = await getStatus(jobId);
        setStatus(s.status);
        setProgress(s.progress || "Working…");
        if (s.status === "completed") {
          setData(await getResults(jobId));
        } else if (s.status === "failed") {
          setError(s.error || "Pipeline failed.");
        } else {
          timer = setTimeout(poll, 3000);
        }
      } catch {
        setError("Cannot reach backend.");
      }
    };
    poll();
    return () => clearTimeout(timer);
  }, [jobId]);

  const flatQ = data ? flattenQuestions(data.questions) : [];
  const filtered = search
    ? flatQ.filter(q =>
        q.label.toLowerCase().includes(search.toLowerCase()) ||
        q.text.toLowerCase().includes(search.toLowerCase())
      )
    : flatQ;

  const activeQ      = activeIndex !== null ? filtered[activeIndex] : null;
  const activeCoords = activeQ && data
    ? lookupCoords(data.segments, activeQ.path)
    : [];
  const targetPage   = activeCoords[0]?.page ?? null;

  // ── Loading ──────────────────────────────────────────────────────────────
  if (status !== "completed" && !error) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center gap-5">
        <Loader2 className="w-12 h-12 text-violet-400 animate-spin" />
        <p className="text-white font-semibold">{progress}</p>
        <p className="text-slate-500 text-sm">OCR can take 3–6 minutes per document…</p>
        <div className="w-64 h-1 rounded-full bg-slate-800 overflow-hidden">
          <div className="h-full w-2/3 bg-violet-500 rounded-full animate-pulse" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="bg-red-950/30 border border-red-700/40 rounded-2xl p-8 text-center max-w-sm">
          <p className="text-white font-semibold text-lg mb-2">Processing Failed</p>
          <p className="text-slate-400 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  // ── Dashboard ────────────────────────────────────────────────────────────
  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-violet-950 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="flex items-center gap-3 px-5 py-3 border-b border-slate-800/60 shrink-0">
        <h1 className="text-xl font-black text-white tracking-tight">
          Veda<span className="text-violet-400">AI</span>
        </h1>
        <span className="text-slate-700">|</span>
        <span className="text-xs text-slate-500 font-mono">{jobId.slice(0,8)}…</span>
        <div className="ml-auto flex items-center gap-3">
          <span className="rounded-full bg-emerald-900/30 border border-emerald-800/40 text-emerald-400 text-xs px-3 py-1">
            {flatQ.length} questions
          </span>
          <a href="/" className="text-xs text-slate-500 hover:text-slate-300 transition-colors">
            ← New upload
          </a>
        </div>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden gap-3 p-3">

        {/* LEFT — scrollable question list */}
        <div className="w-72 shrink-0 flex flex-col gap-2 overflow-hidden">
          {/* Search */}
          <div className="relative shrink-0">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
            <input
              value={search}
              onChange={e => { setSearch(e.target.value); setActiveIndex(null); }}
              placeholder="Search questions…"
              className="w-full bg-slate-800/60 border border-slate-700/50 rounded-xl pl-8 pr-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-violet-600"
            />
          </div>

          {/* Cards */}
          <div className="flex-1 overflow-y-auto flex flex-col gap-1.5 pr-0.5">
            {filtered.map((q, i) => (
              <QuestionCard
                key={q.path}
                q={q}
                isActive={activeIndex === i}
                coords={data ? lookupCoords(data.segments, q.path) : []}
                onClick={() => setActiveIndex(i === activeIndex ? null : i)}
              />
            ))}
            {filtered.length === 0 && (
              <p className="text-xs text-slate-500 text-center mt-8">No questions match your search.</p>
            )}
          </div>
        </div>

        {/* RIGHT — PDF viewer */}
        <div className="flex-1 min-w-0 overflow-hidden">
          {data && (
            <PdfViewer
              jobId={jobId}
              coords={activeCoords}
              targetPage={targetPage}
              pageDims={data.page_dims}
            />
          )}
        </div>
      </div>
    </div>
  );
}
