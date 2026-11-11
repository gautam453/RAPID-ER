import re

for fname in ['journal_main.tex', 'main.tex']:
    p = f'C:/Users/gauta/.gemini/antigravity/scratch/RAPID_ER/manuscript/{fname}'
    with open(p, 'r', encoding='utf-8') as f:
        text = f.read()

    # Revert all {,}
    text = text.replace('{,}', ',')

    # Replace only inside math mode $...$
    def fix_math(m):
        c = m.group(1)
        c = re.sub(r'(\d+),(\d{3})', r'\1{,}\2', c)
        return f'${c}$'

    text = re.sub(r'\$([^$\n]+)\$', fix_math, text)

    with open(p, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Processed {fname}')
