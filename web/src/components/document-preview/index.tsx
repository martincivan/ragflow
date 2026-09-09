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

import { memo, useMemo } from 'react';

import CSVFileViewer from './csv-preview';
import { DocPreviewer } from './doc-preview';
import { EpubPreviewer } from './epub-preview';
import { ExcelCsvPreviewer } from './excel-preview';
import { ImagePreviewer } from './image-preview';
import { Md } from './md';
import PdfPreviewer, { IProps } from './pdf-preview';
import { PptPreviewer } from './ppt-preview';
import { PreviewKind, resolvePreviewKind } from './preview-kind';
import { TxtPreviewer } from './txt-preview';
import { UnsupportedPreview } from './unsupported-preview';
import { VideoPreviewer } from './video-preview';

type PreviewProps = {
  fileType: string;
  className?: string;
  url: string;
  positions?: number[][];
  /** Used by the unsupported-type fallback to name the file and the download. */
  fileName?: string;
};

const DocumentPreview = function ({
  fileType,
  className,
  fileName,
  highlights,
  setWidthAndHeight,
  url,
  positions,
}: PreviewProps & Partial<IProps>) {
  const kind = useMemo(() => resolvePreviewKind(fileType), [fileType]);

  switch (kind) {
    case PreviewKind.Pdf:
      return (
        <section className="h-full">
          <PdfPreviewer
            className={className}
            highlights={highlights}
            setWidthAndHeight={setWidthAndHeight}
            url={url}
          ></PdfPreviewer>
        </section>
      );
    case PreviewKind.Doc:
      return (
        <section>
          <DocPreviewer className={className} url={url} />
        </section>
      );
    case PreviewKind.Text:
      return (
        <section>
          <TxtPreviewer className={className} url={url} />
        </section>
      );
    case PreviewKind.Image:
      return (
        <section>
          <ImagePreviewer className={className} url={url} />
        </section>
      );
    case PreviewKind.Video:
      return (
        <section>
          <VideoPreviewer className={className} url={url} />
        </section>
      );
    case PreviewKind.Ppt:
      return (
        <section>
          <PptPreviewer className={className} url={url} />
        </section>
      );
    case PreviewKind.Excel:
      return (
        <section className="h-full">
          <ExcelCsvPreviewer
            className={className}
            url={url}
            positions={positions}
          />
        </section>
      );
    case PreviewKind.Csv:
      return (
        <section>
          <CSVFileViewer className={className} url={url} />
        </section>
      );
    case PreviewKind.Markdown:
      return (
        <section>
          <Md className={className} url={url} />
        </section>
      );
    case PreviewKind.Epub:
      return (
        <section>
          <EpubPreviewer className={className} url={url} />
        </section>
      );
    default:
      return (
        <section className="h-full">
          <UnsupportedPreview
            className={className}
            fileType={fileType}
            fileName={fileName}
            url={url}
          />
        </section>
      );
  }
};
export default memo(DocumentPreview);
