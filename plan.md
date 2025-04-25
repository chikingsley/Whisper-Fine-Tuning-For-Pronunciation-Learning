1 What to borrow from fine-tuning-whisper-v1.1.0.ipynb

Feature	Why it helps pronunciation FT	How to graft it onto Whisper-small-fine-tuning.ipynb
Stride-overlap decoding during training evaluation	Reduces edge-effects on 30 s chunks ⇒ slightly better token timestamps (which MFA will refine).	Copy the decoding_args = dict(chunk_length=30, stride_length=5) block and pass it to processor.decode(…, **decoding_args) in the eval cell.
Faster-Whisper / CT2 export cell	Gives you an int-4 model that loads < 1 GB and runs 2-3 × faster on CPU/GPU/Mobile.	The pronunciation notebook already pushes to HF; just append the cell:!ct2-transformers-converter --model $repo --output_dir ct2 --quantization int4 --copy_files tokenizer.json
Model-name dropdown	Lets you train small, medium, turbo, etc. without editing code.	Replace the fixed model_id = "openai/whisper-small" with:model_id = widgets.Dropdown(options=["openai/whisper-small","openai/whisper-medium","openai/whisper-turbo"], value="openai/whisper-small") and feed model_id.value into AutoModel….

5 Minimal to-do list
	1.	Merge the three v1.1.0 cells (stride, CT2 export, model dropdown) into Whisper-small-fine-tuning.ipynb.
	2.	Record or acquire ~1-2 h of EN/FR/ES clip pairs (word lists + short phrases).
	3.	Run notebook → get ct2/ int-4 model folder.
	4.	mfa train acoustic data/ my_lang.acoustic and mfa train dictionary data/ my_lang.dict.
	5.	In React Native macOS bind:
	•	whisper_ct2.transcribe()
	•	mfa.force_align()
	•	compute_gop() (20-line Python function).

