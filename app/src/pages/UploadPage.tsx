import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { api, type ProcessConfig } from '@/api/client';
import { Upload, FileSpreadsheet, Loader2 } from 'lucide-react';

export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [domains, setDomains] = useState<string[]>(['hardware']);
  const [config, setConfig] = useState<ProcessConfig>({
    domain: 'hardware',
    workers: 4,
    db_path: 'cache/masks.db',
    result_db_path: 'cache/result.db',
  });

  useEffect(() => {
    api.domains().then(d => setDomains(d.domains)).catch(() => {});
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const uploadRes = await api.upload(file);
      const jobId = uploadRes.job_id;
      await api.process(jobId, config);
      navigate(`/results/${jobId}`);
    } catch (err: any) {
      alert('Ошибка: ' + err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">ENS Verification</h1>
        <p className="text-muted-foreground">
          Загрузите Excel-файл с номенклатурой для обработки и сопоставления с ЕНС
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileSpreadsheet className="w-5 h-5" />
            Загрузка файла
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center hover:border-primary/50 transition-colors">
            <Input
              type="file"
              accept=".xlsx,.xls,.xlsm"
              onChange={handleFileChange}
              className="hidden"
              id="file-input"
            />
            <label htmlFor="file-input" className="cursor-pointer flex flex-col items-center gap-2">
              <Upload className="w-10 h-10 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">
                {file ? file.name : 'Нажмите для выбора Excel-файла (.xlsx, .xls)'}
              </span>
            </label>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Настройки обработки</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="domain">Домен</Label>
              <Select
                value={config.domain}
                onValueChange={v => setConfig(c => ({ ...c, domain: v }))}
              >
                <SelectTrigger id="domain">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {domains.map(d => (
                    <SelectItem key={d} value={d}>{d}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="workers">Потоки</Label>
              <Select
                value={String(config.workers)}
                onValueChange={v => setConfig(c => ({ ...c, workers: Number(v) }))}
              >
                <SelectTrigger id="workers">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[1, 2, 4, 6, 8, 12, 16].map(w => (
                    <SelectItem key={w} value={String(w)}>{w}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="db-path">БД масок</Label>
              <Input
                id="db-path"
                value={config.db_path}
                onChange={e => setConfig(c => ({ ...c, db_path: e.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="result-db-path">БД результатов</Label>
              <Input
                id="result-db-path"
                value={config.result_db_path}
                onChange={e => setConfig(c => ({ ...c, result_db_path: e.target.value }))}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Button
        className="w-full"
        size="lg"
        onClick={handleUpload}
        disabled={!file || uploading}
      >
        {uploading ? (
          <>
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Загрузка и обработка...
          </>
        ) : (
          <>
            <Upload className="w-4 h-4 mr-2" />
            Обработать
          </>
        )}
      </Button>
    </div>
  );
}
