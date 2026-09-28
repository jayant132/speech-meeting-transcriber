
## Run 3: Real 3-Speaker Multilingual Meeting (meeting_recording.mp3, ~80s) — primary deliverable sample

This is the project's primary sample meeting recording, structured as a real discussion (a production bug report, investigation, and resolution decision) rather than a disconnected language test.

- Diarization: correctly identified 3 distinct speakers with accurate, chronologically ordered turns across the full 80-second recording.
- English (Person 1): clean, accurate transcription throughout, confidence 0.69-0.76.
- Hindi (Person 2): clean, accurate transcription including natural code-switched English terms (pplication log, API, code), confidence 0.77-0.83.
- Odia (Person 3): 4 of 6 segments correctly identified and routed to IndicWav2Vec, producing accurate Odia-script output with confidence 0.95-0.97. 2 shorter segments (under 5s) were misclassified by MMS-LID and fell back to Whisper, producing lower-quality output at confidence ~0.74-0.75 (see docs/limitations.md).
- Speaker statistics: correctly calculated speaking time and percentage for all 3 participants (23.44% / 37.67% / 38.89%), summing to 100%.
- Summary: accurately synthesized a real production incident across all 3 speakers and all 3 languages into a single coherent English summary, correctly extracting 4 real decisions (investigate the cause, decide on safest resolution, review code/configuration, maintain system stability) and 3 real action items — the strongest summarization result observed during testing, consistent with the input being a genuine structured discussion rather than disconnected statements.
