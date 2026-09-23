import { RAGFlowFormItem } from '@/components/ragflow-form';
import { SwitchFormField } from '@/components/switch-fom-field';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { MultiSelect } from '@/components/ui/multi-select';
import { useFetchKnowledgeMetadataKeys } from '@/hooks/use-knowledge-request';
import { runChunkMetadataBackfill } from '@/services/knowledge-service';
import { useMutation } from '@tanstack/react-query';
import { useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { useParams, useSearchParams } from 'react-router';
import { toast } from 'sonner';

const FIELD = 'parser_config.chunk_metadata';

export function ChunkMetadataFormField() {
  const { t } = useTranslation();
  const form = useFormContext();
  const { id } = useParams();
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get('id') || id || '';

  const enabled: boolean = useWatch({
    control: form.control,
    name: `${FIELD}.enabled`,
  });
  const ready: boolean = useWatch({
    control: form.control,
    name: `${FIELD}.ready`,
  });
  const fieldsValue: string[] | undefined = useWatch({
    control: form.control,
    name: `${FIELD}.fields`,
  });
  const fields = useMemo(() => fieldsValue ?? [], [fieldsValue]);

  const kbIds = useMemo(() => (datasetId ? [datasetId] : []), [datasetId]);
  const { data: metadataKeys } = useFetchKnowledgeMetadataKeys(kbIds);
  const options = useMemo(() => {
    // Keep already-selected keys visible even if no document currently carries them.
    const keys = new Set([...(metadataKeys ?? []), ...fields]);
    return [...keys].sort().map((key) => ({ label: key, value: key }));
  }, [metadataKeys, fields]);

  const backfill = useMutation({
    mutationFn: async () => {
      const { data } = await runChunkMetadataBackfill(datasetId);
      if (data?.code !== 0) {
        throw new Error(data?.message || 'backfill failed');
      }
      return data;
    },
    onSuccess: () =>
      toast.success(t('knowledgeConfiguration.chunkMetadataBackfillQueued')),
    onError: (e: Error) => toast.error(e.message),
  });

  return (
    <div className="space-y-5">
      <SwitchFormField
        name={`${FIELD}.enabled`}
        label={t('knowledgeConfiguration.chunkMetadataEnabled')}
        tooltip={t('knowledgeConfiguration.chunkMetadataTip')}
        vertical={false}
      />
      {enabled && (
        <>
          <RAGFlowFormItem
            name={`${FIELD}.fields`}
            label={t('knowledgeConfiguration.chunkMetadataFields')}
            tooltip={t('knowledgeConfiguration.chunkMetadataFieldsTip')}
          >
            {(field) => (
              <MultiSelect
                options={options}
                value={field.value ?? []}
                onValueChange={field.onChange}
                placeholder={t('common.pleaseSelect')}
                maxCount={6}
                modalPopover
              />
            )}
          </RAGFlowFormItem>
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm">
              <span>{t('knowledgeConfiguration.chunkMetadataStatus')}</span>
              <Badge variant={ready ? 'default' : 'secondary'}>
                {ready
                  ? t('knowledgeConfiguration.chunkMetadataReady')
                  : t('knowledgeConfiguration.chunkMetadataNotReady')}
              </Badge>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={
                !datasetId ||
                fields.length === 0 ||
                form.formState.isDirty ||
                backfill.isPending
              }
              onClick={() => backfill.mutate()}
            >
              {t('knowledgeConfiguration.chunkMetadataBackfill')}
            </Button>
          </div>
          {form.formState.isDirty && (
            <p className="text-xs text-text-secondary">
              {t('knowledgeConfiguration.chunkMetadataSaveFirst')}
            </p>
          )}
        </>
      )}
    </div>
  );
}
