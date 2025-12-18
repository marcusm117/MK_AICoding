# Project Rules for AI Coding Tools Market Research

## Citation Style

All citations/source links should use the `\citelink{url}{text}` command, which displays the text with a green frame/border to visually indicate it's a clickable citation.

**Usage:**
```latex
\citelink{https://example.com/leaderboard}{Benchmark Name}
```

**Output:** Text surrounded by a green border, clickable to the source URL.

**Example from Table 1:**
```latex
\citelink{https://arcprize.org/leaderboard}{ARC-AGI 1}
```

## File Structure

- `report.tex` - Main document
- `Sections/` - Section files (0_overview.tex, 1_P1M1.tex, etc.)
- `Tables/` - Table files (P1M1-top3.tex, etc.)
- `refs.bib` - Bibliography file

## Language

All content must be in English. The document uses `ctex` package with `scheme=plain` for font support only.

## Writing Style for Slides

**IMPORTANT: Use bullet points/numbered lists with paragraph titles for clear logical structure. Each bullet should contain detailed reasoning with complete, professional sentences and sound citations.**

### Core Principles:

1. **Structure with bullet points + paragraph titles** - Use `\item \textbf{Title:}` to create clear logical sections within each slide. Each bullet point should be a self-contained argument with 2-4 complete sentences.

2. **Detailed reasoning, not fragments** - Do not write "Best for X" or "Leads in Y". Write complete sentences that explain WHY something is true, with evidence and logical connections.

3. **Sound citations throughout** - Every data point, benchmark score, or factual claim should link to its source using `\citelink{url}{text}`. Citations add credibility and allow readers to verify claims.

4. **Quantify all claims** - Use specific numbers with context: "achieves \textcolor{red}{57.8\%} on Terminal-Bench 2.0 (leading by 3.8 percentage points)" not "performs well."

5. **Connect evidence to conclusions** - Each bullet should follow the pattern: [Claim] + [Evidence with citations] + [Implication/Reasoning].

### Formatting Requirements:

1. **`\textbf{Paragraph Title:}`** - Start each bullet point with a bold title that summarizes the point (e.g., "Market Leadership:", "Generalization Weakness:", "6--12 Month Outlook:").

2. **`\textbf{}`** - Also use for model names, key concepts, and terms that should stand out within sentences.

3. **`\textcolor{red}{}`** - Use for critical numbers, key metrics, and important takeaways that readers must not miss.

4. **`\citelink{url}{text}`** - Use for ALL benchmark names, data sources, company pages, and external references.

5. **`\small`** - Standard font size for slide content (never use `\scriptsize` or `\tiny`).

6. **Em-dashes `---`** - Use to add explanatory clauses or parenthetical information.

### Bad Example (fragments, no structure, no citations):
```latex
Best for web development. Leads Terminal-Bench at 57.8%. MCP ecosystem adoption. Good for mainstream tasks.
```

### Bad Example (wall of text, no bullet structure):
```latex
Claude has maintained market leadership since Claude 3.7. The model currently leads Terminal-Bench at 57.8% and SWE-Bench at 74.4%. However, Claude exhibits performance degradation on generalization tests with a 38.3% drop from Verified to Pro. This suggests optimization for familiar scenarios.
```
