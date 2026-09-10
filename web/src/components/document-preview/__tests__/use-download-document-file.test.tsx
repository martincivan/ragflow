import { act, renderHook, waitFor } from '@testing-library/react';
import axios from 'axios';

import message from '@/components/ui/message';
import { downloadFileFromBlob } from '@/utils/file-util';
import { useDownloadDocumentFile } from '../hooks';

jest.mock('@js-preview/excel', () => ({
  __esModule: true,
  default: { init: jest.fn() },
}));
jest.mock('ahooks', () => ({
  useDebounceFn: () => ({ run: jest.fn() }),
  useSize: () => ({ width: 800, height: 600 }),
}));
jest.mock('axios', () => ({ get: jest.fn() }));
jest.mock('jszip');
jest.mock('xlsx', () => ({ read: jest.fn(), write: jest.fn() }));
jest.mock('@/hooks/route-hook', () => ({
  useGetKnowledgeSearchParams: () => ({}),
}));
jest.mock('@/pages/dataflow-result/hooks', () => ({
  useGetPipelineResultSearchParams: () => ({}),
}));
jest.mock('@/utils/api', () => ({ default: {}, restAPIv1: '' }));
jest.mock('@/utils/authorization-util', () => ({
  getAuthorization: () => 'Bearer token-123',
}));
jest.mock('@/constants/authorization', () => ({
  Authorization: 'Authorization',
}));
jest.mock('@/utils/file-util', () => ({ downloadFileFromBlob: jest.fn() }));
jest.mock('@/components/ui/message', () => ({
  __esModule: true,
  default: { error: jest.fn(), success: jest.fn(), warning: jest.fn() },
}));
jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const mockedGet = axios.get as jest.Mock;
const mockedDownload = downloadFileFromBlob as jest.Mock;
const mockedError = (message as unknown as { error: jest.Mock }).error;

const Url = 'http://example.test/api/v1/documents/doc-1/preview';

describe('useDownloadDocumentFile', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // jsdom's console.error is noisy for the deliberate failure case.
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    (console.error as jest.Mock).mockRestore();
  });

  it('sends the Authorization header and saves under the document name', async () => {
    mockedGet.mockResolvedValue({
      data: new ArrayBuffer(4),
      headers: { 'content-type': 'application/epub+zip' },
    });

    const { result } = renderHook(() =>
      useDownloadDocumentFile(Url, 'quarterly report.epub'),
    );

    await act(async () => {
      await result.current.downloadDocumentFile();
    });

    expect(mockedGet).toHaveBeenCalledWith(Url, {
      headers: { Authorization: 'Bearer token-123' },
      responseType: 'arraybuffer',
    });
    const [blob, name] = mockedDownload.mock.calls[0];
    expect(name).toBe('quarterly report.epub');
    expect(blob.type).toBe('application/epub+zip');
    expect(mockedError).not.toHaveBeenCalled();
  });

  it('falls back to a binary content type when the response has no header', async () => {
    mockedGet.mockResolvedValue({ data: new ArrayBuffer(4), headers: {} });

    const { result } = renderHook(() => useDownloadDocumentFile(Url, 'raw'));

    await act(async () => {
      await result.current.downloadDocumentFile();
    });

    expect(mockedDownload.mock.calls[0][0].type).toBe(
      'application/octet-stream',
    );
  });

  it('reports a failure instead of failing silently', async () => {
    mockedGet.mockRejectedValue(new Error('403'));

    const { result } = renderHook(() => useDownloadDocumentFile(Url, 'a.zip'));

    await act(async () => {
      await result.current.downloadDocumentFile();
    });

    expect(mockedDownload).not.toHaveBeenCalled();
    expect(mockedError).toHaveBeenCalledWith('common.downloadFailed');
    await waitFor(() => expect(result.current.downloading).toBe(false));
  });

  it('does nothing without a url', async () => {
    const { result } = renderHook(() => useDownloadDocumentFile('', 'a.zip'));

    await act(async () => {
      await result.current.downloadDocumentFile();
    });

    expect(mockedGet).not.toHaveBeenCalled();
  });
});
