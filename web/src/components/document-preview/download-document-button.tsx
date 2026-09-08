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

import { Button, ButtonProps } from '@/components/ui/button';
import { Download } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useDownloadDocumentFile } from './hooks';

type DownloadDocumentButtonProps = {
  /** Same authenticated file URL the previewers fetch. */
  url: string;
  /** Name the saved file gets; falls back to the browser default when absent. */
  fileName?: string;
  className?: string;
  variant?: ButtonProps['variant'];
  size?: ButtonProps['size'];
  /** Render the label next to the icon. Off for compact header placements. */
  showLabel?: boolean;
};

export function DownloadDocumentButton({
  url,
  fileName,
  className,
  variant = 'outline',
  size = 'default',
  showLabel = true,
}: DownloadDocumentButtonProps) {
  const { t } = useTranslation();
  const { downloadDocumentFile, downloading } = useDownloadDocumentFile(
    url,
    fileName,
  );

  const label = t('common.download');

  return (
    <Button
      variant={variant}
      size={size}
      className={className}
      onClick={downloadDocumentFile}
      loading={downloading}
      disabled={!url}
      title={label}
      aria-label={label}
    >
      <Download className="size-4" />
      {showLabel && <span>{label}</span>}
    </Button>
  );
}
