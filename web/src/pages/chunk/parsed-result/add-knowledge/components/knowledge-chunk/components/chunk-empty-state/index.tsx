/*
 *  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

import { RunningStatus } from '@/constants/knowledge';
import { IKnowledgeFile } from '@/interfaces/database/dataset';
import { CircleAlert, FileSearch, Loader } from 'lucide-react';
import { useTranslation } from 'react-i18next';

type ChunkEmptyStateProps = {
  documentInfo?: Partial<IKnowledgeFile>;
  /** A search string or availability/id filter is narrowing the list. */
  filtered: boolean;
};

/**
 * Rendered instead of an empty chunk list. The chunk-list response already
 * carries the document's parse state, so the page can say *why* there is
 * nothing to show without issuing another request.
 */
export function ChunkEmptyState({
  documentInfo,
  filtered,
}: ChunkEmptyStateProps) {
  const { t } = useTranslation();

  if (filtered) {
    return (
      <EmptyPanel icon={<FileSearch className="size-8 text-text-disabled" />}>
        <p className="text-text-secondary">{t('common.noResults')}</p>
      </EmptyPanel>
    );
  }

  const run = documentInfo?.run;
  const progressMsg = documentInfo?.progress_msg?.trim();
  const percent = Math.max(0, Math.round((documentInfo?.progress ?? 0) * 100));

  if (run === RunningStatus.UNSTART) {
    return (
      <EmptyPanel icon={<FileSearch className="size-8 text-text-disabled" />}>
        <p className="text-text-primary">{t('chunk.emptyNotParsed')}</p>
        <p className="text-text-secondary text-sm">
          {t('chunk.emptyNotParsedTip')}
        </p>
      </EmptyPanel>
    );
  }

  if (
    run === RunningStatus.RUNNING ||
    run === RunningStatus.SCHEDULE ||
    run === RunningStatus.QUEUED
  ) {
    return (
      <EmptyPanel
        icon={<Loader className="size-8 text-text-disabled animate-spin" />}
      >
        <p className="text-text-primary">
          {t('chunk.emptyParsing', { progress: percent })}
        </p>
        <p className="text-text-secondary text-sm">
          {t('chunk.emptyParsingTip')}
        </p>
        <ParsingLog message={progressMsg} />
      </EmptyPanel>
    );
  }

  if (run === RunningStatus.FAIL || run === RunningStatus.CANCEL) {
    return (
      <EmptyPanel icon={<CircleAlert className="size-8 text-state-error" />}>
        <p className="text-text-primary">
          {run === RunningStatus.FAIL
            ? t('chunk.emptyParseFailed')
            : t('chunk.emptyParseCancelled')}
        </p>
        <ParsingLog message={progressMsg} />
      </EmptyPanel>
    );
  }

  return (
    <EmptyPanel icon={<FileSearch className="size-8 text-text-disabled" />}>
      <p className="text-text-primary">{t('chunk.emptyNoChunksProduced')}</p>
      <p className="text-text-secondary text-sm">
        {t('chunk.emptyNoChunksProducedTip')}
      </p>
      <ParsingLog message={progressMsg} />
    </EmptyPanel>
  );
}

function EmptyPanel({
  icon,
  children,
}: React.PropsWithChildren<{ icon: React.ReactNode }>) {
  return (
    <div
      data-testid="chunk-empty-state"
      className="h-full flex flex-col items-center justify-center gap-2 p-6 text-center"
    >
      {icon}
      {children}
    </div>
  );
}

function ParsingLog({ message }: { message?: string }) {
  const { t } = useTranslation();
  if (!message) return null;

  return (
    <details className="mt-2 w-full max-w-2xl text-left">
      <summary className="cursor-pointer text-text-secondary text-sm">
        {t('chunk.parsingLog')}
      </summary>
      <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap rounded border-0.5 border-border-button bg-bg-card p-3 text-xs text-text-secondary">
        {message}
      </pre>
    </details>
  );
}
