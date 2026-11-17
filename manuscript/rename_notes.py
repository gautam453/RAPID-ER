import zipfile
from pathlib import Path

base = Path('C:/Users/gauta/.gemini/antigravity/scratch/RAPID_ER')
for f in ['main.tex', 'journal_main.tex']:
    p = base / 'manuscript' / f
    t = p.read_text(encoding='utf-8')
    t = t.replace(r'\textit{Table notes.}', r'\textit{Brief overview.}')
    p.write_text(t, encoding='utf-8')
    print(f, t.count('Brief overview.'), t.count('Table notes'))

files = [(base / 'manuscript/main.tex', 'main.tex'), (base / 'manuscript/references.bib', 'references.bib')]
for i in range(1, 10):
    for m in (base / 'figures').glob(f'figure{i}_*.png'):
        files.append((m, 'figures/' + m.name))
for z in [base / 'manuscript/overleaf_journal_story.zip', base / 'manuscript/overleaf_project.zip',
          Path('C:/Users/gauta/Downloads/overleaf_journal_story.zip'), Path('C:/Users/gauta/Downloads/overleaf_project.zip')]:
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for s, a in files:
            zf.write(s, a)
print('zips rebuilt')
