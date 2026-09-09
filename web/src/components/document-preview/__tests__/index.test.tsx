import { render, screen } from '@testing-library/react';
import React from 'react';

import DocumentPreview from '..';

jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

// Each previewer is replaced by a probe so the spec asserts on the dispatch,
// not on the (heavy: pdf.js, epub.js, exceljs) previewer implementations.
const probe = (testId: string) => ({
  __esModule: true,
  default: () => React.createElement('div', { 'data-testid': testId }),
});
const namedProbe = (name: string, testId: string) => ({
  __esModule: true,
  [name]: () => React.createElement('div', { 'data-testid': testId }),
});

jest.mock('../pdf-preview', () => probe('pdf-previewer'));
jest.mock('../csv-preview', () => probe('csv-previewer'));
jest.mock('../doc-preview', () => namedProbe('DocPreviewer', 'doc-previewer'));
jest.mock('../epub-preview', () =>
  namedProbe('EpubPreviewer', 'epub-previewer'),
);
jest.mock('../excel-preview', () =>
  namedProbe('ExcelCsvPreviewer', 'excel-previewer'),
);
jest.mock('../image-preview', () =>
  namedProbe('ImagePreviewer', 'image-previewer'),
);
jest.mock('../md', () => namedProbe('Md', 'md-previewer'));
jest.mock('../ppt-preview', () => namedProbe('PptPreviewer', 'ppt-previewer'));
jest.mock('../txt-preview', () => namedProbe('TxtPreviewer', 'txt-previewer'));
jest.mock('../video-preview', () =>
  namedProbe('VideoPreviewer', 'video-previewer'),
);
jest.mock('../hooks', () => ({
  useDownloadDocumentFile: () => ({
    downloadDocumentFile: jest.fn(),
    downloading: false,
  }),
}));

const AllProbes = [
  'pdf-previewer',
  'csv-previewer',
  'doc-previewer',
  'epub-previewer',
  'excel-previewer',
  'image-previewer',
  'md-previewer',
  'ppt-previewer',
  'txt-previewer',
  'video-previewer',
  'unsupported-preview',
];

function renderPreview(fileType: string, fileName?: string) {
  return render(
    React.createElement(DocumentPreview, {
      fileType,
      fileName,
      url: 'http://example.test/api/v1/documents/doc-1/preview',
    }),
  );
}

function expectOnly(testId: string) {
  expect(screen.getByTestId(testId)).toBeInTheDocument();
  AllProbes.filter((id) => id !== testId).forEach((id) => {
    expect(screen.queryByTestId(id)).toBeNull();
  });
}

describe('DocumentPreview dispatch', () => {
  it.each([
    ['pdf', 'pdf-previewer'],
    ['docx', 'doc-previewer'],
    ['txt', 'txt-previewer'],
    ['png', 'image-previewer'],
    ['mp4', 'video-previewer'],
    ['pptx', 'ppt-previewer'],
    ['xlsx', 'excel-previewer'],
    ['csv', 'csv-previewer'],
    ['md', 'md-previewer'],
    ['epub', 'epub-previewer'],
    ['py', 'txt-previewer'],
    ['sql', 'txt-previewer'],
  ])('routes %s to exactly one previewer', (fileType, testId) => {
    renderPreview(fileType);
    expectOnly(testId);
  });

  it.each(['html', 'eml', 'rtf', 'odt', 'zip'])(
    'shows the fallback for %s instead of an empty panel',
    (fileType) => {
      renderPreview(fileType, `report.${fileType}`);
      expectOnly('unsupported-preview');
      expect(screen.getByText(`report.${fileType}`)).toBeInTheDocument();
      expect(
        screen.getByRole('button', { name: 'common.download' }),
      ).toBeInTheDocument();
    },
  );

  it('shows the fallback for a document whose name has no extension', () => {
    const { container } = renderPreview('', 'invoice-scan');
    expectOnly('unsupported-preview');
    expect(container).not.toBeEmptyDOMElement();
    expect(screen.getByText('common.previewNotAvailable')).toBeInTheDocument();
  });
});
