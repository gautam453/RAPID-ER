from pathlib import Path
import zipfile, shutil, re

base = Path('C:/Users/gauta/.gemini/antigravity/scratch/RAPID_ER')
m = base / 'manuscript'

def sub(text, old, new, count=1):
    assert text.count(old) >= 1, 'NOT FOUND: ' + old[:70]
    return text.replace(old, new)

# ---------------- main.tex ----------------
t = (m / 'main.tex').read_text(encoding='utf-8')

# renumber later citations first (descending)
t = sub(t, r'\cite{r11}', r'\cite{r13}')
t = sub(t, r'\cite{r10}', r'\cite{r12}')
t = sub(t, r'\cite{r9}', r'\cite{r11}')
t = sub(t, r'\cite{r7,r8}', r'\cite{r9,r10}')

# abstract discordance wording
t = sub(t, r'contribute to a 5--10\% diagnostic discordance rate',
        r'contribute to measurable diagnostic discordance (up to roughly 10\% between laboratories)')

# intro paragraph 1
old_p1 = t[t.index('Therapeutic decision-making in clinical oncology'):t.index('Clinical practice guidelines from the American Society')]
new_p1 = (r"Therapeutic decision-making in clinical oncology relies heavily on accurate biomarker assessment. For the vast majority of breast cancer patients, the primary molecular determinant guiding clinical management is estrogen receptor alpha (ER$\alpha$), encoded by the \textit{ESR1} gene on chromosome 6q25.1 \cite{r1}. Tumors expressing ER$\alpha$ depend on estrogen-driven transcriptional programs, making them responsive to endocrine therapies such as selective estrogen receptor modulators (tamoxifen), selective estrogen receptor degraders (fulvestrant), and aromatase inhibitors (letrozole, anastrozole). In ER-positive disease, five years of adjuvant tamoxifen reduced breast-cancer mortality by roughly one-third over 15 years of follow-up, with little benefit in ER-negative disease \cite{r2}." + "\n\n")
t = t.replace(old_p1, new_p1)

# intro paragraph 2 discordance + consequences
t = sub(t, r'contribute to a global diagnostic discordance or borderline ambiguity rate of 5--10\% \cite{r4}.',
        r'contribute to measurable inter-laboratory discordance; for example, local and central IHC agreed on ER status in 90\% of cases in the ECOG 2197 trial \cite{r4}.')
t = sub(t, r'while facing delays in necessary cytotoxic chemotherapy \cite{r4}.',
        r'while facing delays in necessary cytotoxic chemotherapy \cite{r2,r3}.')

# intro paragraph 3 multigene assays + ER inference
old_p3 = t[t.index('High-throughput transcriptomic profiling provides a quantitative'):t.index('Instead, this work addresses')]
new_p3 = (r"High-throughput transcriptomic profiling provides a quantitative, reproducible complement to visual histopathology. Clinical multigene expression assays---including Oncotype DX \cite{r5}, MammaPrint \cite{r6}, and Prosigna (PAM50) \cite{r7}---demonstrate that transcriptomic profiles measured on microarray and RNA-based platforms can be translated into clinically actionable classifications. Because \textit{ESR1} expression closely tracks ER status and underpins parsimonious molecular subtype models \cite{r8}, our objective is not merely to show that gene expression separates ER-positive from ER-negative tumors." + "\n\n")
t = t.replace(old_p3, new_p3)

# limitations wording
t = sub(t, r'baseline inter-laboratory variability (5--10\%)', r'baseline inter-laboratory variability (up to roughly 10\%)')

# bibliography
new_bib = r"""\begin{thebibliography}{99}

\bibitem{r1} Menasce LP, White GR, Harrison CJ, Boyle JM. Localization of the estrogen receptor locus (ESR) to chromosome 6q25.1 by FISH and a simple post-FISH banding technique. \textit{Genomics}. 1993;17(1):263--265.

\bibitem{r2} Early Breast Cancer Trialists' Collaborative Group (EBCTCG). Relevance of breast cancer hormone receptors and other factors to the efficacy of adjuvant tamoxifen: patient-level meta-analysis of randomised trials. \textit{Lancet}. 2011;378(9793):771--784.

\bibitem{r3} Allison KH, Hammond MEH, Dowsett M, et al. Estrogen and Progesterone Receptor Testing in Breast Cancer: ASCO/CAP Guideline Update. \textit{J Clin Oncol}. 2020;38(12):1346--1366.

\bibitem{r4} Badve SS, Baehner FL, Gray RP, et al. Estrogen- and progesterone-receptor status in ECOG 2197: comparison of immunohistochemistry by local and central laboratories and quantitative reverse transcription polymerase chain reaction by Oncotype DX. \textit{J Clin Oncol}. 2008;26(15):2473--2481.

\bibitem{r5} Paik S, Shak S, Tang G, et al. A multigene assay to predict recurrence of tamoxifen-treated, node-negative breast cancer. \textit{N Engl J Med}. 2004;351(27):2817--2826.

\bibitem{r6} van 't Veer LJ, Dai H, van de Vijver MJ, et al. Gene expression profiling predicts clinical outcome of breast cancer. \textit{Nature}. 2002;415(6871):530--536.

\bibitem{r7} Parker JS, Mullins M, Cheang MCU, et al. Supervised risk predictor of breast cancer based on intrinsic subtypes. \textit{J Clin Oncol}. 2009;27(8):1160--1167.

\bibitem{r8} Haibe-Kains B, Desmedt C, Loi S, et al. A three-gene model to robustly identify breast cancer molecular subtypes. \textit{J Natl Cancer Inst}. 2012;104(4):311--325.

\bibitem{r9} Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. \textit{BMC Med}. 2019;17(1):230.

\bibitem{r10} Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. \textit{Proc Int Conf Mach Learn (ICML)}. 2017;70:1321--1330.

\bibitem{r11} Leek JT, Scharpf RB, Bravo HC, et al. Tackling the widespread and critical impact of batch effects in high-throughput data. \textit{Nat Rev Genet}. 2010;11(10):733--739.

\bibitem{r12} Meinshausen N, B\"uhlmann P. Stability selection. \textit{J R Stat Soc Series B Stat Methodol}. 2010;72(4):417--473.

\bibitem{r13} Geifman Y, El-Yaniv R. Selective classification for deep neural networks. \textit{Adv Neural Inf Process Syst (NeurIPS)}. 2017;30:4878--4887.

\end{thebibliography}"""
i0 = t.index(r'\begin{thebibliography}')
i1 = t.index(r'\end{thebibliography}') + len(r'\end{thebibliography}')
t = t[:i0] + new_bib + t[i1:]

# citation integrity
cites = set(k.strip() for c in re.findall(r'\\cite\{([^}]+)\}', t) for k in c.split(','))
items = set(re.findall(r'\\bibitem\{([^}]+)\}', t))
print('cited:', sorted(cites, key=lambda x: int(x[1:])))
print('missing:', cites - items, 'unused:', items - cites)
(m / 'main.tex').write_text(t, encoding='utf-8')
shutil.copyfile(m / 'main.tex', m / 'journal_main.tex')

# ---------------- references.bib: preserve user's markdown, restore real BibTeX ----------------
bib = m / 'references.bib'
old = bib.read_text(encoding='utf-8')
if old.lstrip().startswith('#'):
    (m / 'manuscript_revised.md').write_text(old, encoding='utf-8')
    print('saved markdown from references.bib to manuscript_revised.md')

entries = [
 ('menasce1993esr', 'article', dict(author="Menasce, L. P. and White, G. R. and Harrison, C. J. and Boyle, J. M.", title="Localization of the estrogen receptor locus ({ESR}) to chromosome 6q25.1 by {FISH} and a simple post-{FISH} banding technique", journal="Genomics", year="1993", volume="17", number="1", pages="263--265")),
 ('ebctcg2011relevance', 'article', dict(author="{Early Breast Cancer Trialists' Collaborative Group (EBCTCG)}", title="Relevance of breast cancer hormone receptors and other factors to the efficacy of adjuvant tamoxifen: patient-level meta-analysis of randomised trials", journal="Lancet", year="2011", volume="378", number="9793", pages="771--784")),
 ('allison2020estrogen', 'article', dict(author="Allison, Kimberly H. and Hammond, M. Elizabeth H. and Dowsett, Mitchell and others", title="Estrogen and Progesterone Receptor Testing in Breast Cancer: {ASCO/CAP} Guideline Update", journal="Journal of Clinical Oncology", year="2020", volume="38", number="12", pages="1346--1366")),
 ('badve2008ecog', 'article', dict(author="Badve, Sunil S. and Baehner, Frederick L. and Gray, Robert P. and others", title="Estrogen- and progesterone-receptor status in {ECOG} 2197: comparison of immunohistochemistry by local and central laboratories and quantitative reverse transcription polymerase chain reaction by {Oncotype DX}", journal="Journal of Clinical Oncology", year="2008", volume="26", number="15", pages="2473--2481")),
 ('paik2004multigene', 'article', dict(author="Paik, Soonmyung and Shak, Steven and Tang, Gong and others", title="A multigene assay to predict recurrence of tamoxifen-treated, node-negative breast cancer", journal="New England Journal of Medicine", year="2004", volume="351", number="27", pages="2817--2826")),
 ('vantveer2002gene', 'article', dict(author="van 't Veer, Laura J. and Dai, Hongyue and van de Vijver, Marc J. and others", title="Gene expression profiling predicts clinical outcome of breast cancer", journal="Nature", year="2002", volume="415", number="6871", pages="530--536")),
 ('parker2009supervised', 'article', dict(author="Parker, Joel S. and Mullins, Michael and Cheang, Maggie C. U. and others", title="Supervised risk predictor of breast cancer based on intrinsic subtypes", journal="Journal of Clinical Oncology", year="2009", volume="27", number="8", pages="1160--1167")),
 ('haibekains2012three', 'article', dict(author="Haibe-Kains, Benjamin and Desmedt, Christine and Loi, Sherene and others", title="A three-gene model to robustly identify breast cancer molecular subtypes", journal="Journal of the National Cancer Institute", year="2012", volume="104", number="4", pages="311--325")),
 ('vancalster2019calibration', 'article', dict(author="Van Calster, Ben and McLernon, David J. and van Smeden, Maarten and Wynants, Laure and Steyerberg, Ewout W.", title="Calibration: the {Achilles} heel of predictive analytics", journal="BMC Medicine", year="2019", volume="17", number="1", pages="230")),
 ('guo2017calibration', 'inproceedings', dict(author="Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.", title="On calibration of modern neural networks", booktitle="Proceedings of the 34th International Conference on Machine Learning (ICML)", series="Proceedings of Machine Learning Research", volume="70", pages="1321--1330", year="2017")),
 ('leek2010tackling', 'article', dict(author="Leek, Jeffrey T. and Scharpf, Robert B. and Bravo, H{\\'e}ctor Corrada and others", title="Tackling the widespread and critical impact of batch effects in high-throughput data", journal="Nature Reviews Genetics", year="2010", volume="11", number="10", pages="733--739")),
 ('meinshausen2010stability', 'article', dict(author="Meinshausen, Nicolai and B{\\\"u}hlmann, Peter", title="Stability selection", journal="Journal of the Royal Statistical Society: Series B (Statistical Methodology)", year="2010", volume="72", number="4", pages="417--473")),
 ('geifman2017selective', 'inproceedings', dict(author="Geifman, Yonatan and El-Yaniv, Ran", title="Selective classification for deep neural networks", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", volume="30", pages="4878--4887", year="2017")),
]
out = []
for key, typ, f in entries:
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items())
    out.append(f"@{typ}{{{key},\n{body}\n}}\n")
bib.write_text("\n".join(out), encoding='utf-8')
print('wrote BibTeX with', len(entries), 'entries')

# ---------------- zips ----------------
files = [(m / 'main.tex', 'main.tex')]
for p in sorted((base / 'figures').glob('figure[1-9]_*.png')):
    files.append((p, 'figures/' + p.name))
for z in [m / 'overleaf_journal_story.zip', m / 'overleaf_project.zip',
          Path('C:/Users/gauta/Downloads/overleaf_journal_story.zip'), Path('C:/Users/gauta/Downloads/overleaf_project.zip')]:
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for s, a in files:
            zf.write(s, a)
print('zips rebuilt')
