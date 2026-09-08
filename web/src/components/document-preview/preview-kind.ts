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

import { Images } from '@/constants/common';

export enum PreviewKind {
  Pdf = 'pdf',
  Doc = 'doc',
  Text = 'text',
  Image = 'image',
  Video = 'video',
  Ppt = 'ppt',
  Excel = 'excel',
  Csv = 'csv',
  Markdown = 'markdown',
  Epub = 'epub',
  /** No in-browser previewer exists for this file; offer the file itself instead. */
  Unsupported = 'unsupported',
}

const DocExtensions = ['doc', 'docx'];

/**
 * Extensions the plain-text previewer can render as-is. Beyond `txt`/`json`
 * these mirror the source files the backend already serves as `text/plain`
 * (see `CONTENT_TYPE_MAP` in `api/utils/file_response.py`).
 *
 * Markup that the backend deliberately forces to `attachment` for security
 * (html, xhtml, xml, svg, mhtml) is intentionally absent: those download
 * rather than preview.
 */
const TextExtensions = [
  'txt',
  'json',
  'log',
  'c',
  'conf',
  'cpp',
  'cs',
  'go',
  'h',
  'ini',
  'java',
  'js',
  'jsx',
  'kt',
  'php',
  'py',
  'rb',
  'rs',
  'sh',
  'sql',
  'toml',
  'ts',
  'tsx',
  'yaml',
  'yml',
];

const VideoExtensions = [
  'mp4',
  'avi',
  'mov',
  'mkv',
  'wmv',
  'flv',
  'mpeg',
  'mpg',
  'asf',
  'rm',
  'rmvb',
];

const PptExtensions = ['ppt', 'pptx'];

const ExcelExtensions = ['xlsx', 'xls'];

const MarkdownExtensions = ['md', 'mdx', 'markdown'];

const KindByExtension = new Map<string, PreviewKind>([
  ...DocExtensions.map((ext) => [ext, PreviewKind.Doc] as const),
  ...TextExtensions.map((ext) => [ext, PreviewKind.Text] as const),
  ...Images.map((ext) => [ext, PreviewKind.Image] as const),
  ...VideoExtensions.map((ext) => [ext, PreviewKind.Video] as const),
  ...PptExtensions.map((ext) => [ext, PreviewKind.Ppt] as const),
  ...ExcelExtensions.map((ext) => [ext, PreviewKind.Excel] as const),
  ...MarkdownExtensions.map((ext) => [ext, PreviewKind.Markdown] as const),
  ['pdf', PreviewKind.Pdf],
  ['csv', PreviewKind.Csv],
  ['epub', PreviewKind.Epub],
]);

/**
 * Map a file extension onto the single previewer that should render it.
 * Anything unrecognised — including an empty string, which is what a document
 * whose name carries no extension resolves to — is `Unsupported` so the caller
 * can show a fallback instead of an empty panel.
 */
export function resolvePreviewKind(fileType: string | undefined): PreviewKind {
  const normalized = (fileType ?? '').trim().toLowerCase().replace(/^\./, '');
  return KindByExtension.get(normalized) ?? PreviewKind.Unsupported;
}
