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

import classNames from 'classnames';
import { FileQuestion } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { DownloadDocumentButton } from './download-document-button';

type UnsupportedPreviewProps = {
  fileType: string;
  fileName?: string;
  url: string;
  className?: string;
};

/**
 * Shown when no in-browser previewer can render the file. Without this the
 * panel rendered nothing at all, leaving no way to tell an unsupported type
 * apart from a failed load and no way to reach the file itself.
 */
export function UnsupportedPreview({
  fileType,
  fileName,
  url,
  className,
}: UnsupportedPreviewProps) {
  const { t } = useTranslation();
  const extension = fileType?.trim().toUpperCase();

  return (
    <div
      data-testid="unsupported-preview"
      className={classNames(
        'w-full h-full flex flex-col items-center justify-center gap-3 p-6 text-center',
        className,
      )}
    >
      <FileQuestion className="size-10 text-text-disabled" />
      {fileName && (
        <p className="max-w-full truncate text-text-primary font-medium">
          {fileName}
        </p>
      )}
      <p className="text-text-secondary text-sm">
        {extension
          ? t('common.previewNotAvailableForType', { type: extension })
          : t('common.previewNotAvailable')}
      </p>
      <p className="text-text-disabled text-xs max-w-80">
        {t('common.previewNotAvailableTip')}
      </p>
      <DownloadDocumentButton url={url} fileName={fileName} />
    </div>
  );
}
