# Lekka: brand system

## Name
**Lekka** (లెక్క) is Telugu for *the count, the math, the reckoning*. It's the thing most of us skip when we re-pick last year's health plan. Say it "LEK-kah".
- **Tagline:** *Do the math before you click "same as last year".*
- **Short line:** *Your plan, worked out.*

## Voice
- Plain, calm, specific. Short sentences. Numbers over adjectives.
- Talk like a friend who is good with money, not like an insurer or an app.
- Say what it doesn't know: "a typical year", "a bad year (1 in 10)", "check this".
- **Never use:** revolutionary, seamless, effortless, AI-powered, unlock, supercharge, game-changer, emoji in UI copy.

## Mark
- `mark.svg`: a **tally of five**. Four strokes and a stroke across: the oldest way people count, and a line through "same as last year".
- `mark-mono.svg` uses `currentColor`. `favicon.svg` is the same file as the mark.
- Minimum size 16 px. Keep clear space of a quarter of the mark's width around it. Never rotate it, add gradients or add shadows.

## Colour

| Token | Hex | Use |
|---|---|---|
| `--paper` | `#F5F0E6` | Page background (warm, like a ledger page) |
| `--ink` | `#1E2521` | Text, strokes |
| `--ledger` | `#2E6B57` | Good outcome, primary action, the cross stroke |
| `--rust` | `#B5482E` | Bad-year cost, warnings, "check this" |
| `--highlight` | `#F2D16B` | **Quote highlight**: every number's source line is shown as if marked with a highlighter |
| `--mute` | `#6F7671` | Secondary text, gridlines |
| `--card` | `#FFFDF8` | Cards on paper |
| `--line` | `#E3DCCD` | Borders, rules |

**Dark mode:**
- paper `#151A17`, card `#1C221F`, ink `#ECE7DC`
- ledger `#5BA88C`, rust `#E07A5F`
- highlight `#8A7322` (the text on it uses ink)

Contrast: ink on paper and ink on card are both ≥ 12:1. White on ledger is ≥ 5:1.

## Type
- **Headings:** *Newsreader* (Google Fonts), 600, slightly tight tracking.
- **Body and UI:** *IBM Plex Sans*, 400 and 500.
- **Numbers, quotes and money:** *IBM Plex Mono*, tabular figures. Money always uses tabular numerals so columns line up.

## Signature components
1. **The quote highlight.** Any number can be tapped to show its source line from the plan PDF, highlighted in `--highlight`, with the page number. This is the product's promise made visible.
2. **The result card:**
   - one sentence on top: *"Plan B costs you about $1,140 less in a typical year."*
   - Below it, two bars per plan: a solid bar for a typical year and an outlined bar for a bad year, in ledger and rust.
3. **"Check this" states.** Rust outline with plain text explaining what couldn't be confirmed. Never a silent guess.
4. **The tally of plans.** Plans are listed as rows, like lines in a ledger, not glossy pricing tiers.

## Layout
- A paper background with cards. Generous margins, max content width 720 px for reading and 1080 px for comparison tables.
- Works on a 360 px phone. Tables become stacked rows.
- Motion: 150–200 ms fades and slides only. No parallax, no confetti.

## Don't
- Gradients, glassmorphism, neon, stock medical icons (stethoscopes, hearts, pills), or robots and sparkles.
- Insurer blue.
- Saying "AI" in the product UI. The models are named honestly in the README and the post.
