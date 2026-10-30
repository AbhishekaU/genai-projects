"""
The video's script: one entry per slide, in order. Everything else in video/
reads from this list, so editing a line here and re-running make_video.py is
all it takes to change what's said (only the changed slide's audio is
re-generated).

    narration -- what the voice says (spelled for the ear, e.g. "Kwad")
    caption   -- what the subtitles show; defaults to the narration
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Slide:
    id: str
    narration: str
    caption: str | None = None

    @property
    def subtitle(self) -> str:
        return self.caption or self.narration


SLIDES = [
    Slide("01_hook",
          "Most company data is locked inside documents. This is a pipeline that reads "
          "commercial contracts, and proves how well it does it."),
    Slide("02_problem",
          "A raw contract PDF goes in. A clean JSON record comes out: parties, dates, "
          "governing law, and nine types of clauses."),
    Slide("03_architecture",
          "Gemini extracts every field with structured output. Each value carries a "
          "word-for-word quote, and the pipeline checks that the quote really exists."),
    Slide("04_dataset",
          "I scored it against Kwad: five hundred and ten real contracts labelled by "
          "lawyers, with precision and recall for every field.",
          caption="I scored it against CUAD: 510 real contracts labelled by lawyers, "
                  "with precision and recall for every field."),
    Slide("05_journey",
          "Five prompt versions. Every change came from reading the errors, not guessing, "
          "and the final version was tested on contracts it had never seen."),
    Slide("06_thinking",
          "First lesson: lowering Gemini's thinking level made it three times faster, "
          "with a third of the output tokens, at almost the same accuracy."),
    Slide("07_arithmetic",
          "Second: don't make the model do arithmetic. It extracts the term length, and "
          "code computes the expiration date. Recall jumped from forty six to seventy seven percent.",
          caption="Second: don't make the model do arithmetic. It extracts the term length, "
                  "and code computes the expiration date. Recall jumped from 46% to 77%."),
    Slide("08_definitions",
          "Third: the model wasn't reading my field definitions. Moving them into the "
          "prompt took cap on liability from fifty to one hundred percent recall.",
          caption="Third: the model wasn't reading my field definitions. Moving them into "
                  "the prompt took cap on liability from 50% to 100% recall."),
    Slide("09_grounding",
          "Every quote is checked against the source text. Ninety nine percent or more "
          "are verified, and some errors turned out to be gaps in the lawyers' labels.",
          caption="Every quote is checked against the source text. 99% or more are "
                  "verified, and some errors turned out to be gaps in the lawyers' labels."),
    Slide("10_parsers",
          "Text extraction, or Gemini reading the PDF directly: the same accuracy. But one "
          "invisible character was inflating the bill. Fixing it cut input tokens by twenty six percent.",
          caption="Text extraction, or Gemini reading the PDF directly: the same accuracy. "
                  "But one invisible character was inflating the bill. Fixing it cut input tokens by 26%."),
    Slide("11_demo",
          "In the app, you upload a contract and get validated JSON back in about seven seconds."),
    Slide("12_results",
          "On held-out contracts: eighty eight percent on key terms, and ninety two on "
          "clauses. The code and the full evaluation are on GitHub. Project one of six.",
          caption="On held-out contracts: 88% on key terms, and 92% on clauses. The code "
                  "and the full evaluation are on GitHub. Project 1 of 6."),
]
