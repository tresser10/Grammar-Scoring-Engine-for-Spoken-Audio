# Grammar-Scoring-Engine-for-Spoken-Audio
Goal: predict a continuous MOS-Likert grammar score (0-5) from a 45-60 s speech clip.

Approach in one paragraph. Grammar is a property of what was said, so we first convert speech to text with Whisper (ASR), then describe each clip with three complementary feature families: (1) text/grammar features (GPT-2 fluency/perplexity, LanguageTool error rates, sentence statistics, fillers, repetitions), (2) semantic sentence embeddings of the transcript, and (3) acoustic/prosodic features (pause statistics, speech rate, ASR confidence, wav2vec2 embeddings). Several regressors are trained with K-fold cross-validation, ensembled, and evaluated with RMSE and Pearson correlation. The final report (metrics, plots, discussion) is at the bottom of the notebook.

Note: However this training is based on cpu and not gpu since the device i was using did not have an integrated gpu. Also when trying a gpu based training in colab it showed me various errors related to the libraries hence execution was very complex. However in cpu the training takes a lot of time so gpu version are to better to look upon. 
