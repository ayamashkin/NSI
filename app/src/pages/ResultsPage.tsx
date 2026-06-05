import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
// Select components available for future filters
// import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { api, type FilterRequest, type JobStatus } from '@/api/client';
import { ArrowLeft, Loader2, RefreshCw, Filter, Download, FileJson, CheckCircle, XCircle } from 'lucide-react';
import VerifyModal from '@/components/VerifyModal';

interface ResultItem {
  id: number;
  text: string;
  ens_code?: string;
  ens_name?: string;
  success: boolean;
  confidence: number;
  match_type_ru?: string;
  item_type?: string;
  standard?: string;
  params?: Record<string, any>;
  details?: Record<string, any>;
}

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [job, setJob] = useState<JobStatus | null>(null);
  const [results, setResults] = useState<ResultItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [polling, setPolling] = useState(true);
  const [verifyIdx, setVerifyIdx] = useState<number | null>(null);

  const [filters, setFilters] = useState<FilterRequest>({
    standard: '',
    item_type: '',
    confidence_min: undefined,
    confidence_max: undefined,
    success_only: false,
    limit: 50,
    offset: 0,
  });

  const loadJob = useCallback(async () => {
    if (!jobId) return;
    try {
      const j = await api.jobStatus(jobId);
      setJob(j);
      if (j.status === 'completed' || j.status === 'failed') {
        setPolling(false);
      }
    } catch {
      setPolling(false);
    }
  }, [jobId]);

  const loadResults = useCallback(async () => {
    if (!jobId || job?.status !== 'completed') return;
    setLoading(true);
    try {
      const cleanFilters = {
        ...filters,
        standard: filters.standard || undefined,
        item_type: filters.item_type || undefined,
        confidence_min: filters.confidence_min || undefined,
        confidence_max: filters.confidence_max || undefined,
      };
      const res = await api.results(jobId, cleanFilters);
      setResults(res.results || []);
      setTotal(res.total || 0);
    } catch (err: any) {
      console.error('Failed to load results:', err);
    } finally {
      setLoading(false);
    }
  }, [jobId, job?.status, filters]);

  // Poll job status
  useEffect(() => {
    if (!polling) return;
    loadJob();
    const interval = setInterval(loadJob, 2000);
    return () => clearInterval(interval);
  }, [polling, loadJob]);

  // Load results when job completes
  useEffect(() => {
    if (job?.status === 'completed') {
      loadResults();
    }
  }, [job?.status, loadResults]);

  const handleVerify = (idx: number) => {
    setVerifyIdx(idx);
  };

  const handleVerified = () => {
    setVerifyIdx(null);
    loadResults();
  };

  const getConfidenceColor = (c: number) => {
    if (c >= 0.9) return 'bg-green-100 text-green-800';
    if (c >= 0.7) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getMatchTypeColor = (type?: string) => {
    if (!type) return 'secondary';
    if (type.includes('Точное')) return 'default';
    if (type.includes('подстановки')) return 'secondary';
    if (type.includes('Нечеткое')) return 'outline';
    return 'secondary';
  };

  if (!job) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (job.status === 'processing') {
    const progress = job.progress;
    const percent = progress?.percent || 0;
    return (
      <div className="max-w-xl mx-auto p-6 space-y-6">
        <Button variant="ghost" onClick={() => navigate('/')} className="gap-2">
          <ArrowLeft className="w-4 h-4" /> Назад
        </Button>
        <Card>
          <CardContent className="p-8 text-center space-y-4">
            <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto" />
            <div className="space-y-2">
              <h2 className="text-xl font-semibold">Обработка...</h2>
              <p className="text-muted-foreground">
                {progress?.current || 0} / {progress?.total || job.rows || '?'} строк
              </p>
            </div>
            <div className="w-full bg-muted rounded-full h-2">
              <div
                className="bg-primary h-2 rounded-full transition-all"
                style={{ width: `${percent}%` }}
              />
            </div>
            {progress?.stats && (
              <div className="flex justify-center gap-4 text-sm">
                <span className="text-green-600">✓ {progress.stats.success || 0}</span>
                <span className="text-red-600">✗ {progress.stats.failed || 0}</span>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  if (job.status === 'failed') {
    return (
      <div className="max-w-xl mx-auto p-6 space-y-6">
        <Button variant="ghost" onClick={() => navigate('/')} className="gap-2">
          <ArrowLeft className="w-4 h-4" /> Назад
        </Button>
        <Card className="border-red-200">
          <CardContent className="p-8 text-center space-y-4">
            <XCircle className="w-12 h-12 text-red-500 mx-auto" />
            <h2 className="text-xl font-semibold text-red-600">Ошибка обработки</h2>
            <p className="text-muted-foreground">{job.error}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => navigate('/')} className="gap-2">
            <ArrowLeft className="w-4 h-4" /> Назад
          </Button>
          <div>
            <h1 className="text-2xl font-bold">Результаты</h1>
            <p className="text-sm text-muted-foreground">
              {job.filename} · {job.rows} строк · {job.stats?.success || 0} успешно · {job.stats?.failed || 0} ошибок
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => api.export(jobId!, 'excel')}>
            <Download className="w-4 h-4 mr-1" /> Excel
          </Button>
          <Button variant="outline" size="sm" onClick={() => api.export(jobId!, 'json')}>
            <FileJson className="w-4 h-4 mr-1" /> JSON
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-2 mb-3">
            <Filter className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm font-medium">Фильтры</span>
          </div>
          <div className="grid grid-cols-5 gap-3">
            <div>
              <Label className="text-xs">Стандарт</Label>
              <Input
                size={1}
                placeholder="ОСТ 1..."
                value={filters.standard}
                onChange={e => setFilters(f => ({ ...f, standard: e.target.value }))}
              />
            </div>
            <div>
              <Label className="text-xs">Тип изделия</Label>
              <Input
                placeholder="Болт..."
                value={filters.item_type}
                onChange={e => setFilters(f => ({ ...f, item_type: e.target.value }))}
              />
            </div>
            <div>
              <Label className="text-xs">Confidence min</Label>
              <Input
                type="number"
                min={0}
                max={1}
                step={0.1}
                placeholder="0.0"
                value={filters.confidence_min ?? ''}
                onChange={e => setFilters(f => ({ ...f, confidence_min: e.target.value ? Number(e.target.value) : undefined }))}
              />
            </div>
            <div>
              <Label className="text-xs">Confidence max</Label>
              <Input
                type="number"
                min={0}
                max={1}
                step={0.1}
                placeholder="1.0"
                value={filters.confidence_max ?? ''}
                onChange={e => setFilters(f => ({ ...f, confidence_max: e.target.value ? Number(e.target.value) : undefined }))}
              />
            </div>
            <div className="flex items-end gap-2">
              <Button
                variant={filters.success_only ? 'default' : 'outline'}
                size="sm"
                onClick={() => setFilters(f => ({ ...f, success_only: !f.success_only }))}
              >
                <CheckCircle className="w-3 h-3 mr-1" />
                Только успешные
              </Button>
              <Button size="sm" onClick={loadResults}>
                <RefreshCw className="w-3 h-3 mr-1" />
                Применить
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results Table */}
      {loading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted">
                  <tr>
                    <th className="p-2 text-left">#</th>
                    <th className="p-2 text-left w-[300px]">Наименование</th>
                    <th className="p-2 text-left">Код ЕНС</th>
                    <th className="p-2 text-center">Confidence</th>
                    <th className="p-2 text-left">Тип</th>
                    <th className="p-2 text-left">Стандарт</th>
                    <th className="p-2 text-left">Сопоставление</th>
                    <th className="p-2 text-center">Действия</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <tr key={i} className="border-t hover:bg-muted/50">
                      <td className="p-2 text-muted-foreground">{r.id + 1}</td>
                      <td className="p-2 max-w-[300px] truncate" title={r.text}>{r.text}</td>
                      <td className="p-2 font-mono text-xs">
                        {r.ens_code ? (
                          <span className="text-green-600">{r.ens_code}</span>
                        ) : (
                          <span className="text-red-400">—</span>
                        )}
                      </td>
                      <td className="p-2 text-center">
                        <Badge className={getConfidenceColor(r.confidence)}>
                          {r.confidence?.toFixed(3) || '0.000'}
                        </Badge>
                      </td>
                      <td className="p-2">{r.item_type || '—'}</td>
                      <td className="p-2 text-xs">{r.standard || '—'}</td>
                      <td className="p-2">
                        <Badge variant={getMatchTypeColor(r.match_type_ru)} className="text-xs">
                          {r.match_type_ru || '—'}
                        </Badge>
                      </td>
                      <td className="p-2 text-center">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleVerify(r.id)}
                          disabled={r.confidence >= 1.0}
                        >
                          Верифицировать
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {results.length === 0 && (
              <div className="p-12 text-center text-muted-foreground">
                Нет результатов. Примените другие фильтры.
              </div>
            )}
            <div className="p-3 border-t text-xs text-muted-foreground text-right">
              Показано {results.length} из {total} записей
            </div>
          </CardContent>
        </Card>
      )}

      {verifyIdx !== null && (
        <VerifyModal
          jobId={jobId!}
          resultIdx={verifyIdx}
          onClose={() => setVerifyIdx(null)}
          onVerified={handleVerified}
        />
      )}
    </div>
  );
}
