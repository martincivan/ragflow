import { PreviewKind, resolvePreviewKind } from '../preview-kind';

describe('resolvePreviewKind', () => {
  it.each([
    ['pdf', PreviewKind.Pdf],
    ['doc', PreviewKind.Doc],
    ['docx', PreviewKind.Doc],
    ['txt', PreviewKind.Text],
    ['json', PreviewKind.Text],
    ['png', PreviewKind.Image],
    ['mp4', PreviewKind.Video],
    ['pptx', PreviewKind.Ppt],
    ['xlsx', PreviewKind.Excel],
    ['csv', PreviewKind.Csv],
    ['md', PreviewKind.Markdown],
    ['mdx', PreviewKind.Markdown],
    ['epub', PreviewKind.Epub],
  ])('keeps %s routed to its existing previewer', (ext, kind) => {
    expect(resolvePreviewKind(ext)).toBe(kind);
  });

  it.each(['py', 'js', 'ts', 'java', 'c', 'cpp', 'h', 'php', 'go', 'sh', 'cs', 'kt', 'sql', 'yml'])(
    'renders %s as plain text instead of nothing',
    (ext) => {
      expect(resolvePreviewKind(ext)).toBe(PreviewKind.Text);
    },
  );

  it('adds markdown as a markdown alias', () => {
    expect(resolvePreviewKind('markdown')).toBe(PreviewKind.Markdown);
  });

  it.each(['html', 'htm', 'xml', 'svg', 'eml', 'rtf', 'odt', 'zip'])(
    'falls back for %s rather than rendering a blank panel',
    (ext) => {
      expect(resolvePreviewKind(ext)).toBe(PreviewKind.Unsupported);
    },
  );

  it.each([undefined, '', '   ', 'unknown', 'doc-with-no-extension'])(
    'falls back for %p',
    (value) => {
      expect(resolvePreviewKind(value)).toBe(PreviewKind.Unsupported);
    },
  );

  it('normalizes case and a leading dot', () => {
    expect(resolvePreviewKind('.PDF')).toBe(PreviewKind.Pdf);
    expect(resolvePreviewKind('PNG')).toBe(PreviewKind.Image);
  });
});
