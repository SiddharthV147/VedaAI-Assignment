// lib/api.ts — typed client for the simplified VedaAI backend
import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const api = axios.create({ baseURL: API_BASE });

// ── Status ─────────────────────────────────────────────────────────────────

export interface StatusResponse {
  job_id:   string;
  status:   "pending" | "processing" | "completed" | "failed";
  progress: string | null;
  error:    string | null;
}

// ── Coord entry inside _coords arrays ──────────────────────────────────────

export interface CoordEntry {
  page:  number;
  bbox:  [number, number, number, number]; // [x1, y1, x2, y2] in OCR pixels
}

// ── Results payload ────────────────────────────────────────────────────────
// questions  → raw nested dict from questions.json
// segments   → raw nested dict from answersheet_results.json (segments key)
// page_dims  → {"1": {"width_px": …, "height_px": …}, …}

export interface ResultsPayload {
  job_id:    string;
  questions: Record<string, any>;           // Section → Question → {_text, (a): …}
  segments:  Record<string, any>;           // Section → Question → {_coords: …}
  page_dims: Record<string, { width_px: number; height_px: number }>;
}

// ── API calls ──────────────────────────────────────────────────────────────

export async function uploadFiles(qp: File, ans: File): Promise<string> {
  const form = new FormData();
  form.append("question_paper", qp);
  form.append("answer_sheet",   ans);
  const { data } = await api.post<{ job_id: string }>("/api/v1/upload", form);
  return data.job_id;
}

export async function getStatus(jobId: string): Promise<StatusResponse> {
  const { data } = await api.get<StatusResponse>(`/api/v1/status/${jobId}`);
  return data;
}

export async function getResults(jobId: string): Promise<ResultsPayload> {
  const { data } = await api.get<ResultsPayload>(`/api/v1/results/${jobId}`);
  return data;
}

export function getPdfUrl(jobId: string): string {
  return `${API_BASE}/api/v1/pdf/${jobId}`;
}

// ── Helpers — flatten nested dicts into flat question list ─────────────────

export interface FlatQuestion {
  /** Full path: "Section A.Question 1" or "Section A.Question 1.(a)" */
  path:    string;
  section: string;
  label:   string;   // "Question 1" or "(a)"
  text:    string;
  depth:   number;   // 0 = section, 1 = question, 2 = sub-part
}

function sectionOrder(section: string): [number, string] {
  const letter = section.match(/section\s*([A-Z])/i)?.[1];
  return letter ? [0, letter.toUpperCase()] : [1, section];
}

function questionNumber(label: string): number {
  const match = label.match(/question\s*(\d+)/i) || label.match(/(\d+)/);
  return match ? parseInt(match[1], 10) : Number.MAX_SAFE_INTEGER;
}

function subpartOrder(label: string): number {
  const token = label.match(/^\(\s*([a-z]+|[ivxlcdm]+)\s*\)$/i)?.[1]?.toLowerCase();
  if (!token) return Number.MAX_SAFE_INTEGER;
  if (/^[ivxlcdm]+$/.test(token) && token.length > 1) {
    const values: Record<string, number> = { i: 1, v: 5, x: 10, l: 50, c: 100, d: 500, m: 1000 };
    let n = 0;
    for (let i = 0; i < token.length; i++) {
      const cur = values[token[i]] ?? 0;
      const next = values[token[i + 1]] ?? 0;
      n += cur < next ? -cur : cur;
    }
    return n;
  }
  return token.charCodeAt(0);
}

function compareKeys(a: string, b: string, key: (s: string) => number | [number, string]): number {
  const ka = key(a);
  const kb = key(b);
  if (Array.isArray(ka) && Array.isArray(kb)) {
    if (ka[0] !== kb[0]) return ka[0] - kb[0];
    return ka[1] < kb[1] ? -1 : ka[1] > kb[1] ? 1 : 0;
  }
  return (ka as number) - (kb as number);
}

/** Extract a flat list of questions, sections A–n then questions 1–n. */
export function flattenQuestions(
  questions: Record<string, any>
): FlatQuestion[] {
  const result: FlatQuestion[] = [];
  const sections = Object.entries(questions).sort(([a], [b]) =>
    compareKeys(a, b, sectionOrder)
  );
  for (const [section, qs] of sections) {
    const entries = Object.entries(qs as Record<string, any>)
      .filter(([qKey, qVal]) => typeof qVal === "object" && qKey !== "_instructions")
      .sort(([a], [b]) => questionNumber(a) - questionNumber(b));
    for (const [qKey, qVal] of entries) {
      result.push({
        path:    `${section}.${qKey}`,
        section,
        label:   qKey,
        text:    (qVal as any)._text || "",
        depth:   1,
      });
      const subs = Object.entries(qVal as Record<string, any>)
        .filter(([subKey, subVal]) => subKey.startsWith("(") && typeof subVal === "object")
        .sort(([a], [b]) => subpartOrder(a) - subpartOrder(b));
      for (const [subKey, subVal] of subs) {
        result.push({
          path:    `${section}.${qKey}.${subKey}`,
          section,
          label:   subKey,
          text:    (subVal as any)._text || "",
          depth:   2,
        });
      }
    }
  }
  return result;
}

/**
 * Given a question path ("Section A.Question 1.(a)") look up _coords in the
 * matching place of the segments nested dict.
 */
export function lookupCoords(
  segments: Record<string, any>,
  path: string
): CoordEntry[] {
  const parts = path.split(".");
  // Navigate into segments: parts[0]=section, parts[1]=question, parts[2]=sub
  let node: any = segments;
  for (const part of parts) {
    if (node && typeof node === "object" && part in node) {
      node = node[part];
    } else {
      return [];
    }
  }
  return Array.isArray(node?._coords) ? node._coords : [];
}
