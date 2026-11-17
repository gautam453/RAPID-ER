import re, zipfile, shutil
from pathlib import Path

base = Path('C:/Users/gauta/.gemini/antigravity/scratch/RAPID_ER')
m = base / 'manuscript'
t = (m / 'main.tex').read_text(encoding='utf-8')

# sanity checks
figs = re.findall(r'\\includegraphics\[.*?\]\{([^}]+)\}', t)
print('figures in order:', figs)
caps = re.findall(r'\\label\{(fig:[^}]+)\}', t)
print('fig labels:', caps)
print('tables:', re.findall(r'\\label\{(tab:[^}]+)\}', t))
cites = set(re.findall(r'\\cite\{([^}]+)\}', t))
items = set(re.findall(r'\\bibitem\{([^}]+)\}', t))
used = set(k.strip() for c in cites for k in c.split(','))
print('missing bibitems:', used - items, 'unused:', items - used)
refs = set(re.findall(r'\\ref\{([^}]+)\}', t))
labels = set(re.findall(r'\\label\{([^}]+)\}', t))
print('undefined refs:', refs - labels)
print('begin/end figure', t.count('\\begin{figure}'), t.count('\\end{figure}'), 'table', t.count('\\begin{table}'), t.count('\\end{table}'))

shutil.copyfile(m / 'main.tex', m / 'journal_main.tex')

# consistency in other docs
for f in [m / 'manuscript.md', m / 'supplementary' / 'supplementary_materials.md']:
    s = f.read_text(encoding='utf-8')
    s = s.replace('Kunniyur Ramchander Gautam', 'Gautam Kunniyur Ramchander')
    f.write_text(s, encoding='utf-8')

files = [(m / 'main.tex', 'main.tex')]
for p in sorted((base / 'figures').glob('figure[1-9]_*.png')):
    files.append((p, 'figures/' + p.name))
for z in [m / 'overleaf_journal_story.zip', m / 'overleaf_project.zip',
          Path('C:/Users/gauta/Downloads/overleaf_journal_story.zip'), Path('C:/Users/gauta/Downloads/overleaf_project.zip')]:
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for s_, a in files:
            zf.write(s_, a)
    print('wrote', z, z.stat().st_size)
