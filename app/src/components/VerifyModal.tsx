import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { api, type Candidate } from '@/api/client';
import { X, Loader2, CheckCircle, ArrowRight } from 'lucide-react';

interface Props {
  jobId: string;
  resultIdx: number;
  onClose: () => void;
  onVerified: () => void;
}

export default function VerifyModal({ jobId, resultIdx, onClose, onVerified }: Props) {
  const [loading, setLoading] = useState(true);
  const [text, setText] = useState('');
  const [currentCode, setCurrentCode] = useState('');
  const [currentName, setCurrentName] = useState('');
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [manualCode, setManualCode] = useState('');

  useEffect(() => {
    api.candidates(jobId, resultIdx)
      .then(res => {
        setText(res.text || '');
        setCurrentCode(res.ens_code || '');
        setCurrentName(res.ens_name || '');
        setCandidates(res.candidates || []);
      })
      .catch(err => console.error('Failed to load candidates:', err))
      .finally(() => setLoading(false));
  }, [jobId, resultIdx]);

  const handleVerify = async () => {
    let ensCode = manualCode.trim();
    let ensName = '';

    if (!ensCode && selectedIdx !== null) {
      const selected = candidates[selectedIdx];
      ensCode = selected.ens_code || '';
      ensName = selected.name || '';
    }

    if (!ensCode) {
      alert('Введите код ЕНС или выберите кандидата');
      return;
    }

    setSaving(true);
    try {
      await api.verify(jobId, resultIdx, {
        ens_code: ensCode,
        ens_name: ensName,
        confidence: 1.0,
      });
      onVerified();
    } catch (err: any) {
      alert('Ошибка верификации: ' + err.message);
    } finally {
      setSaving(false);
    }
  };

  const getParamStatus = (status: string) => {
    if (status === 'exact') return 'text-green-600 font-medium';
    if (status === 'exact (in name)') return 'text-blue-600';
    if (status.includes('token')) return 'text-yellow-600';
    return 'text-red-500';
  };

  const getParamIcon = (status: string) => {
    if (status === 'exact') return '=';
    if (status === 'exact (in name)') return '~';
    if (status.includes('token')) return '~';
    return '!=';
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-3xl max-h-[90vh] overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-primary" />
            Верификация сопоставления
          </CardTitle>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </CardHeader>
        <CardContent className="space-y-4 overflow-y-auto max-h-[calc(90vh-80px)]">
          {/* Original text */}
          <div className="bg-muted p-3 rounded-md">
            <div className="text-xs text-muted-foreground mb-1">Исходное наименование</div>
            <div className="font-medium">{text}</div>
          </div>

          {/* Current match */}
          {currentCode && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Текущий ЕНС:</span>
              <Badge variant="outline" className="font-mono">{currentCode}</Badge>
              <span className="text-muted-foreground truncate">{currentName}</span>
            </div>
          )}

          {loading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <>
              {/* Candidates */}
              {candidates.length > 0 ? (
                <div className="space-y-2">
                  <div className="text-sm font-medium">Кандидаты (топ-5)</div>
                  {candidates.map((cd, i) => (
                    <button
                      key={i}
                      onClick={() => { setSelectedIdx(i); setManualCode(''); }}
                      className={`w-full text-left p-3 rounded-md border transition-all ${
                        selectedIdx === i
                          ? 'border-primary bg-primary/5'
                          : 'border-muted hover:border-muted-foreground/50'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground">{i + 1}.</span>
                          <span className="font-mono text-sm">{cd.ens_code || '—'}</span>
                          <ArrowRight className="w-3 h-3 text-muted-foreground" />
                          <span className="text-sm truncate max-w-[300px]">{cd.name || '—'}</span>
                        </div>
                        <Badge className={cd.score >= 0.9 ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}>
                          {cd.score?.toFixed(3) || '0.000'}
                        </Badge>
                      </div>
                      {/* Params comparison */}
                      {cd.params_comparison && Object.keys(cd.params_comparison).length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-2 text-xs">
                          {Object.entries(cd.params_comparison).map(([pk, pv]: [string, any]) => (
                            <span key={pk} className={getParamStatus(pv.status || '')}>
                              {pk}: {pv.extracted || '?'} {getParamIcon(pv.status || '')} {pv.ens_value || '?'}
                            </span>
                          ))}
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground py-4 text-center">
                  Кандидаты не найдены. Введите код ЕНС вручную.
                </div>
              )}

              {/* Manual input */}
              <div className="space-y-2 pt-2 border-t">
                <div className="text-sm font-medium">Или введите код ЕНС вручную</div>
                <div className="flex gap-2">
                  <Input
                    placeholder="1000000000"
                    value={manualCode}
                    onChange={e => { setManualCode(e.target.value); setSelectedIdx(null); }}
                  />
                </div>
              </div>

              {/* Actions */}
              <div className="flex justify-end gap-2 pt-2">
                <Button variant="outline" onClick={onClose}>Отмена</Button>
                <Button onClick={handleVerify} disabled={saving}>
                  {saving ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <CheckCircle className="w-4 h-4 mr-2" />
                  )}
                  Подтвердить
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
