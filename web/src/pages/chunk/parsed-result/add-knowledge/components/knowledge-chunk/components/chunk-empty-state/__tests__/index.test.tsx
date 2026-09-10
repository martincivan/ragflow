import { render, screen } from '@testing-library/react';
import React from 'react';

import { RunningStatus } from '@/constants/knowledge';
import { ChunkEmptyState } from '..';

jest.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) =>
      options ? `${key}:${JSON.stringify(options)}` : key,
  }),
}));

function renderState(
  documentInfo: Record<string, unknown>,
  filtered = false,
) {
  return render(
    React.createElement(ChunkEmptyState, {
      documentInfo: documentInfo as never,
      filtered,
    }),
  );
}

describe('ChunkEmptyState', () => {
  it('explains an unparsed document', () => {
    renderState({ run: RunningStatus.UNSTART, progress: 0 });
    expect(screen.getByText('chunk.emptyNotParsed')).toBeInTheDocument();
  });

  it('reports parsing progress as a percentage', () => {
    renderState({ run: RunningStatus.RUNNING, progress: 0.42 });
    expect(
      screen.getByText('chunk.emptyParsing:{"progress":42}'),
    ).toBeInTheDocument();
  });

  it('surfaces the parsing log on failure', () => {
    renderState({
      run: RunningStatus.FAIL,
      progress: -1,
      progress_msg: '[ERROR]file type not supported yet',
    });
    expect(screen.getByText('chunk.emptyParseFailed')).toBeInTheDocument();
    expect(
      screen.getByText('[ERROR]file type not supported yet'),
    ).toBeInTheDocument();
  });

  it('distinguishes a finished parse that produced nothing', () => {
    renderState({
      run: RunningStatus.DONE,
      progress: 1,
      progress_msg: 'No chunk built from report.rtf',
    });
    expect(
      screen.getByText('chunk.emptyNoChunksProduced'),
    ).toBeInTheDocument();
    expect(
      screen.getByText('No chunk built from report.rtf'),
    ).toBeInTheDocument();
  });

  it('keeps the plain no-results message while a filter is active', () => {
    renderState({ run: RunningStatus.DONE, progress: 1 }, true);
    expect(screen.getByText('common.noResults')).toBeInTheDocument();
    expect(screen.queryByText('chunk.emptyNoChunksProduced')).toBeNull();
  });
});
