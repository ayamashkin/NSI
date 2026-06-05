const API_BASE = '/api';

async function fetchJson(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status}: ${err}`);
  }
  return res.json();
}

export interface ProcessConfig {
  domain: string;
  workers: number;
  db_path: string;
  result_db_path: string;
}

export interface FilterRequest {
  standard?: string;
  item_type?: string;
  confidence_min?: number;
  confidence_max?: number;
  success_only: boolean;
  limit: number;
  offset: number;
}

export interface VerifyRequest {
  ens_code: string;
  ens_name?: string;
  confidence: number;
}

export interface JobStatus {
  job_id: string;
  status: string;
  filename?: string;
  rows?: number;
  progress?: {
    current: number;
    total: number;
    percent: number;
    stats: Record<string, number>;
  };
  stats?: Record<string, number>;
  error?: string;
}

export interface Candidate {
  ens_code?: string;
  name?: string;
  score: number;
  params_comparison: Record<string, any>;
}

export const api = {
  health: () => fetchJson('/health'),
  domains: () => fetchJson('/domains'),

  upload: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return fetchJson('/upload', { method: 'POST', body: form });
  },

  process: (jobId: string, config: ProcessConfig) =>
    fetchJson(`/process/${jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }),

  jobStatus: (jobId: string) => fetchJson(`/jobs/${jobId}`),

  results: (jobId: string, filters: FilterRequest) =>
    fetchJson(`/jobs/${jobId}/results`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(filters),
    }),

  candidates: (jobId: string, resultIdx: number) =>
    fetchJson(`/jobs/${jobId}/results/${resultIdx}/candidates`),

  verify: (jobId: string, resultIdx: number, req: VerifyRequest) =>
    fetchJson(`/jobs/${jobId}/results/${resultIdx}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),

  export: (jobId: string, format: 'excel' | 'json') => {
    window.open(`${API_BASE}/jobs/${jobId}/export/${format}`, '_blank');
  },

  searchDb: (params: Record<string, any>) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') qs.append(k, String(v));
    });
    return fetchJson(`/result-db/search?${qs.toString()}`);
  },
};
