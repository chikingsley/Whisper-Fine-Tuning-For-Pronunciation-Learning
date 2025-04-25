import argparse
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from pathlib import Path

import torch
import evaluate
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from huggingface_hub import HfApi, HfFolder

# Global variables for processor (feature_extractor and tokenizer)
# These will be set in the main function
feature_extractor: Optional[WhisperFeatureExtractor] = None
tokenizer: Optional[WhisperTokenizer] = None

# --- 3. Data Preparation --- #

def prepare_dataset(batch, text_column_name="sentence", audio_column_name="audio"):
    """Prepares a single batch of audio data for model training."""
    global feature_extractor, tokenizer
    if feature_extractor is None or tokenizer is None:
        raise ValueError("Feature extractor and tokenizer must be initialized before calling prepare_dataset.")

    # compute log-Mel input features from input audio array
    audio = batch[audio_column_name]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch[text_column_name]).input_ids
    return batch

# --- 4. Data Collator --- #

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads the inputs received.

    Args:
        processor ([`Wav2Vec2Processor`]):
            The processor used for processing the data.
        forward_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return attention_mask.
    """

    processor: Any
    forward_attention_mask: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        global tokenizer # Use the global tokenizer

        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# --- 5. Metrics --- #

metric = evaluate.load("wer")

def compute_metrics(pred):
    """Computes Word Error Rate (WER) metric for the model predictions."""
    global tokenizer # Use the global tokenizer

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main(args):
    global feature_extractor, tokenizer # Declare we are modifying the globals

    print("Starting training script with args:", args)
    # --- 1. Login to Hugging Face (if required for private models/datasets or pushing) ---
    if args.push_to_hub or args.hf_token or args.dataset_name.count('/') == 1: # Check if dataset is likely on hub
        token = args.hf_token or HfFolder.get_token()
        if not token and args.dataset_name.count('/') == 1: # Only fail if dataset is on hub and no token
             raise ValueError("Hugging Face token not found. Needed for Hub dataset/model or pushing. Please login using `huggingface-cli login` or pass --hf_token.")
        if token:
            try:
                api = HfApi()
                user_info = api.whoami(token)
                print(f"Logged in as {user_info['name']}")
                if args.final_model_name:
                     hub_model_id = f"{user_info['name']}/{args.final_model_name}"
                else:
                     hub_model_id = None # Cannot push without final name
            except Exception as e:
                print(f"Could not authenticate with token: {e}")
                token = None
                hub_model_id = None
        else: # No token provided or found
             token = None
             hub_model_id = None

    else: # Not pushing and dataset likely local
        token = None
        hub_model_id = None
        if args.push_to_hub:
             print("Warning: --push_to_hub specified but no --final_model_name provided. Will not push.")


    # --- 2. Load Processor (Feature Extractor + Tokenizer) --- #
    print(f"Loading processor for model: {args.model_id}")
    # Load Feature extractor and Tokenizer
    # Assign to global variables
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id, token=token)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_id, language=args.language_full, task="transcribe", token=token)
    # Load Processor which wraps Feature Extractor and Tokenizer
    processor = WhisperProcessor.from_pretrained(args.model_id, language=args.language_full, task="transcribe", token=token)
    print("Processor loaded.")


    # --- 3 & 4. Load and Prepare Dataset & Data Collator --- #
    print(f"Loading dataset configuration: {args.dataset_name}")

    if args.dataset_name.endswith(".csv"):
        print("Detected local CSV dataset path.")
        dataset_path = Path(args.dataset_name)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset CSV file not found at: {dataset_path}")

        # Assume audio folder is in the same directory as the CSV
        audio_folder_path = dataset_path.parent / "audio_folder"
        if not audio_folder_path.is_dir():
             # Fallback: Check relative to current working directory if not next to CSV
             script_dir_audio_path = Path("audio_folder")
             if script_dir_audio_path.is_dir():
                 audio_folder_path = script_dir_audio_path.resolve()
                 print(f"Audio folder found relative to script execution directory: {audio_folder_path}")
             else:
                 raise FileNotFoundError(f"Audio folder 'audio_folder' not found next to CSV ({dataset_path.parent}) or in current directory ({Path.cwd()}).")
        else:
             audio_folder_path = audio_folder_path.resolve() # Get absolute path
             print(f"Audio folder found next to CSV: {audio_folder_path}")


        # Load the dataset from CSV
        # Use the same CSV for train/eval for this small test, split later if needed
        raw_datasets = load_dataset("csv", data_files={"train": str(dataset_path), "eval": str(dataset_path)})

        # Function to create the full audio path
        def create_audio_path(batch):
            # Check if audio column already exists and is not just the filename
            if args.audio_column_name in batch and isinstance(batch[args.audio_column_name], str):
                 batch[args.audio_column_name] = str(audio_folder_path / batch[args.audio_column_name])
            # Check if 'audio' column exists (common default) and is not the target audio col name already processed
            elif 'audio' in batch and isinstance(batch['audio'], str) and args.audio_column_name != 'audio':
                 batch[args.audio_column_name] = str(audio_folder_path / batch['audio'])
                 # We might want to rename the column if audio_column_name is different
                 # but casting usually handles a dict with 'path' key correctly
                 # batch['audio_path'] = str(audio_folder_path / batch['audio'])
            else:
                 # Attempt to construct path if audio_column_name holds the filename directly
                 try:
                      batch[args.audio_column_name] = str(audio_folder_path / batch[args.audio_column_name])
                 except TypeError:
                      print(f"Warning: Could not construct audio path for batch entry: {batch}. Ensure '{args.audio_column_name}' contains filenames.")

            return batch

        print("Mapping audio file names to full paths...")
        # Use remove_columns=False initially if the audio column name needs to be derived
        raw_datasets = raw_datasets.map(create_audio_path, num_proc=1) # Use 1 proc for this simple map

        # Now we can remove columns IF the path mapping worked and audio column is correct
        columns_to_keep = {args.audio_column_name, args.text_column_name}
        columns_to_remove = [col for col in raw_datasets["train"].column_names if col not in columns_to_keep]

        # Ensure we have the audio column after mapping
        if args.audio_column_name not in raw_datasets["train"].column_names:
             raise ValueError(f"Audio column '{args.audio_column_name}' not found after path mapping. Check CSV headers and mapping logic.")

        print(f"Removing columns: {columns_to_remove}")
        raw_datasets = raw_datasets.remove_columns(columns_to_remove)

        # Split train set for evaluation if only one split was loaded
        if "eval" not in raw_datasets:
             print("No eval split found, splitting train set 90/10 for evaluation.")
             # Ensure dataset is large enough to split
             if len(raw_datasets["train"]) < 10:
                  print("Warning: Train set too small to split effectively. Using full set for eval.")
                  raw_datasets["eval"] = raw_datasets["train"]
             else:
                  raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1)


    else: # Load from Hub
        print(f"Loading dataset from Hub: {args.dataset_name}")
        raw_datasets = DatasetDict()
        raw_datasets["train"] = load_dataset(args.dataset_name, split=args.train_split, use_auth_token=token)
        raw_datasets["eval"] = load_dataset(args.dataset_name, split=args.eval_split, use_auth_token=token)
        print(f"Dataset loaded: {raw_datasets}")

        # Remove unnecessary columns from Hub dataset
        columns_to_remove = set(raw_datasets["train"].column_names) - {args.audio_column_name, args.text_column_name}
        # Adjust removal based on actual column names if needed
        if args.audio_column_name in columns_to_remove: columns_to_remove.remove(args.audio_column_name)
        if args.text_column_name in columns_to_remove: columns_to_remove.remove(args.text_column_name)
        print(f"Removing columns: {list(columns_to_remove)}")
        raw_datasets = raw_datasets.remove_columns(list(columns_to_remove))


    print(f"Final dataset structure: {raw_datasets}")

    # Downsample and filter audio (common step for both local and hub)
    print("Casting audio column and downsampling to 16kHz...")
    try:
        raw_datasets = raw_datasets.cast_column(args.audio_column_name, Audio(sampling_rate=16000))
    except Exception as e:
        print(f"Error casting audio column: {e}")
        print("Please ensure the audio column contains valid paths or data.")
        raise

    # Define filter function that works with the 'audio' column directly
    def is_audio_in_length_range(batch):
        """Filter function that checks audio length directly from the audio array"""
        # Get audio array and check if it has data
        try:
            audio = batch[args.audio_column_name]
            # Check if it's already processed (dict with 'array' key)
            if isinstance(audio, dict) and 'array' in audio and 'sampling_rate' in audio:
                audio_length = len(audio['array']) / audio['sampling_rate']
                return args.min_input_length <= audio_length <= args.max_input_length
            # For unprocessed audio paths, we'll return True and filter later if needed
            return True
        except Exception as e:
            print(f"Warning: Could not check audio length for an item: {e}")
            # Keep items we can't check for now
            return True

    max_input_length = args.max_input_length * 16000
    min_input_length = args.min_input_length * 16000

    if args.max_input_length > 0 or args.min_input_length > 0:
        print(f"Filtering dataset for audio length between {args.min_input_length}s and {args.max_input_length}s...")
        try:
            # Filter using our custom function that doesn't require 'input_length'
            raw_datasets = raw_datasets.filter(
                is_audio_in_length_range,
                num_proc=args.dataloader_num_workers  # Use multiple processes if specified
            )
            print(f"Dataset filtered: {raw_datasets}")
        except Exception as e:
            print(f"Warning: Error during audio length filtering: {e}")
            print("Continuing without length filtering...")


    print("Preprocessing dataset...")
    vectorized_datasets = raw_datasets.map(
        prepare_dataset, # Use the standalone function
        remove_columns=raw_datasets["train"].column_names, # Remove original cols after processing
        fn_kwargs={"text_column_name": args.text_column_name, "audio_column_name": args.audio_column_name},
        num_proc=args.dataloader_num_workers # Use multiple processes if specified
    )
    print("Dataset preprocessed.")

    # Instantiate data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


    # --- 5. Load Pre-trained Model & Apply PEFT --- #
    print(f"Loading pre-trained model: {args.model_id}")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        # load_in_8bit=True, # Example: if you wanted 8-bit loading
        device_map="auto", # Automatically distributes model across available devices
        token=token
    )

    # ===> Set Stride-Overlap Decoding <===
    print("Setting stride-overlap generation config...")
    model.generation_config.language = args.language # Use short code for generation
    model.generation_config.task = "transcribe"
    model.generation_config.chunk_length_s = 30
    model.generation_config.stride_length_s = 5
    # Ensure forced_decoder_ids and suppress_tokens are handled correctly
    # (usually defaults are fine, but being explicit can help)
    # model.generation_config.forced_decoder_ids = None # processor handles this based on language/task
    # model.generation_config.suppress_tokens = []

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.config.use_cache = False # Needs to be False for gradient checkpointing
        # Note: PEFT models might require enabling input gradients explicitly if using GC
        # model.gradient_checkpointing_enable() # Often done by Trainer if arg is set

    if args.use_peft:
        print("Applying PEFT/LoRA...")
        # Ensure modules_to_save doesn't contain invalid modules if model architecture changes
        # Example: Add projection layers if needed for a specific Whisper variant or task
        # modules_to_save = ["proj_out"] # Only if training the final projection layer

        config = LoraConfig(
            r=args.peft_r,
            lora_alpha=args.peft_alpha,
            target_modules=args.peft_target_modules,
            lora_dropout=args.peft_dropout,
            bias="none",
            # modules_to_save=modules_to_save # Optional: specify layers to train fully
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    # --- 6. Define Training Arguments --- #
    print("Defining training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        # num_train_epochs=args.num_train_epochs, # Prefer max_steps
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to=args.report_to.split(',') if args.report_to else ["tensorboard"], # Allow comma-separated list
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        # Required for PEFT + gradient_checkpointing compatibility potentially
        # remove_unused_columns=False, # Only if needed, test without first
        # label_names=["labels"],       # Only if needed
    )

    # --- 7. Initialize Trainer --- #
    print("Initializing Trainer...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.predict_with_generate else None, # Only pass if generating
        tokenizer=processor.feature_extractor, # Pass feature extractor for padding inputs
    )

    # Ensure cache is disabled correctly if GC is used with PEFT/non-PEFT
    if args.gradient_checkpointing:
         model.config.use_cache = False

    # --- 8. Training --- #
    if args.do_train:
        print("*** Training ***")
        train_result = trainer.train()
        # trainer.save_model() # Saves PEFT adapters by default if PEFT is used

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state() # Saves optimizer, scheduler, etc.

        # Save the final PEFT adapter model explicitly if needed
        if args.use_peft:
            final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
            model.save_pretrained(final_adapter_dir)
            print(f"Final PEFT adapters saved to {final_adapter_dir}")
        else: # Save full model if not using PEFT
            final_model_dir = os.path.join(args.output_dir, "final_model")
            trainer.save_model(final_model_dir) # Saves full model
            processor.save_pretrained(final_model_dir)
            print(f"Final full model saved to {final_model_dir}")

    # --- 9. Evaluation --- #
    if args.do_eval:
        print("*** Evaluating ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=args.generation_max_length,
            # num_beams=1 # Use greedy decoding for eval? Or default beam search?
        )
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    # --- 10. Merge PEFT Adapters & Save Final Model --- #
    final_model_dir = args.output_dir # Base directory for saving merged model

    if args.use_peft:
        print("*** Merging PEFT adapters and saving final model ***")
        # Determine path to best adapters (or final if not loading best)
        adapter_path = trainer.state.best_model_checkpoint if args.load_best_model_at_end and trainer.state.best_model_checkpoint else os.path.join(args.output_dir, "final_adapter")

        # Check if the adapter path actually exists
        if not os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
             print(f"Warning: Adapter config not found at {adapter_path}. Falling back to potentially last saved checkpoint or final_adapter dir if exists.")
             # Fallback logic: Check last checkpoint or the explicit final adapter save
             checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
             if checkpoints:
                  latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                  adapter_path_fallback = os.path.join(args.output_dir, latest_checkpoint)
                  if os.path.exists(os.path.join(adapter_path_fallback, 'adapter_config.json')):
                       print(f"Using latest checkpoint adapter: {adapter_path_fallback}")
                       adapter_path = adapter_path_fallback
                  elif os.path.exists(os.path.join(args.output_dir, "final_adapter", 'adapter_config.json')):
                      adapter_path = os.path.join(args.output_dir, "final_adapter")
                      print(f"Using final adapter saved at: {adapter_path}")
                  else:
                      raise FileNotFoundError(f"Could not find valid PEFT adapter checkpoint in {args.output_dir} or its subdirectories.")
             elif os.path.exists(os.path.join(args.output_dir, "final_adapter", 'adapter_config.json')):
                 adapter_path = os.path.join(args.output_dir, "final_adapter")
                 print(f"Using final adapter saved at: {adapter_path}")
             else:
                  raise FileNotFoundError(f"Could not find valid PEFT adapter checkpoint in {args.output_dir}. Training might have failed or adapters were not saved correctly.")

        print(f"Loading base model ({args.model_id}) for merging...")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
            device_map="auto",
            token=token,
        )

        print(f"Loading PEFT adapter from: {adapter_path}")
        model_to_merge = PeftModel.from_pretrained(base_model, adapter_path)

        print("Merging adapters...")
        merged_model = model_to_merge.merge_and_unload()
        print("Adapters merged.")

        final_model_dir = os.path.join(args.output_dir, "final_merged_model")
        print(f"Saving merged model to {final_model_dir}...")
        merged_model.save_pretrained(final_model_dir)
        processor.save_pretrained(final_model_dir)
        print("Merged model saved.")
    else: # If not using PEFT, the best model is already saved by Trainer if load_best_model_at_end=True
        if args.load_best_model_at_end and trainer.state.best_model_checkpoint:
            final_model_dir = trainer.state.best_model_checkpoint
            print(f"Using best model saved at: {final_model_dir}")
        elif os.path.exists(os.path.join(args.output_dir, "final_model")): # Fallback if not loading best or no checkpoint found
             final_model_dir = os.path.join(args.output_dir, "final_model")
             print(f"Using final model saved at: {final_model_dir}")
        else: # Fallback to base output dir, hoping trainer saved something there
             final_model_dir = args.output_dir
             print(f"Warning: Could not determine specific best/final model path. Using base output directory: {final_model_dir}")


    # --- 11. CTranslate2 Conversion --- #
    if args.do_ct2_conversion:
        print("*** Converting model to CTranslate2 format ***")
        ct2_output_dir = os.path.join(args.output_dir, "ct2_model")
        # Ensure the final model directory exists
        if not os.path.exists(final_model_dir) or not os.path.exists(os.path.join(final_model_dir, "config.json")):
             print(f"Error: Cannot find model to convert in {final_model_dir}. Skipping CTranslate2 conversion.")
        else:
            cmd = [
                "ct2-transformers-converter",
                "--model", final_model_dir,
                "--output_dir", ct2_output_dir,
                "--quantization", args.ct2_quantization,
                "--force" # Overwrite existing ct2 conversion
            ]
            # Copy tokenizer files manually as ct2 converter might not grab everything needed by processor
            # Check if processor files exist before copying
            if os.path.exists(os.path.join(final_model_dir, "tokenizer.json")):
                 cmd.extend(["--copy_files", "tokenizer.json"])
            if os.path.exists(os.path.join(final_model_dir, "preprocessor_config.json")):
                 cmd.extend(["--copy_files", "preprocessor_config.json"])
            if os.path.exists(os.path.join(final_model_dir, "config.json")):
                 cmd.extend(["--copy_files", "config.json"])
            # Add other files if necessary (e.g., added_tokens.json, special_tokens_map.json)
            if os.path.exists(os.path.join(final_model_dir, "vocab.json")):
                 cmd.extend(["--copy_files", "vocab.json"])
            if os.path.exists(os.path.join(final_model_dir, "merges.txt")):
                 cmd.extend(["--copy_files", "merges.txt"])
            if os.path.exists(os.path.join(final_model_dir, "normalizer.json")):
                 cmd.extend(["--copy_files", "normalizer.json"])
            if os.path.exists(os.path.join(final_model_dir, "special_tokens_map.json")):
                 cmd.extend(["--copy_files", "special_tokens_map.json"])
            if os.path.exists(os.path.join(final_model_dir, "added_tokens.json")):
                 cmd.extend(["--copy_files", "added_tokens.json"])


            print(f"Running CTranslate2 conversion command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"CTranslate2 model saved to {ct2_output_dir}")
            except subprocess.CalledProcessError as e:
                print(f"Error during CTranslate2 conversion:")
                print("Command:", e.cmd)
                print("Return Code:", e.returncode)
                print("Output:", e.stdout)
                print("Error Output:", e.stderr)
            except FileNotFoundError:
                 print("Error: 'ct2-transformers-converter' command not found. Make sure ctranslate2 is installed and in your PATH.")


    # --- 12. Push to Hub --- #
    if args.push_to_hub and hub_model_id:
        print(f"*** Pushing final model to Hub: {hub_model_id} ***")
        try:
            # Need to load the potentially merged model again if PEFT was used
            if args.use_peft:
                 print(f"Loading merged model from {final_model_dir} for push...")
                 model_to_push = WhisperForConditionalGeneration.from_pretrained(final_model_dir, token=token)
                 processor_to_push = WhisperProcessor.from_pretrained(final_model_dir, token=token)
            else: # Load directly from the final saved directory (could be checkpoint or final_model)
                 print(f"Loading final model from {final_model_dir} for push...")
                 model_to_push = WhisperForConditionalGeneration.from_pretrained(final_model_dir, token=token)
                 processor_to_push = WhisperProcessor.from_pretrained(final_model_dir, token=token)

            # Create repo if it doesn't exist
            api.create_repo(hub_model_id, exist_ok=True, token=token)

            print("Pushing model and processor...")
            model_to_push.push_to_hub(hub_model_id, token=token)
            processor_to_push.push_to_hub(hub_model_id, token=token)

            print(f"Model pushed successfully to {hub_model_id}")

            # Optionally push CT2 model as well
            if args.do_ct2_conversion and os.path.exists(ct2_output_dir):
                print(f"Pushing CT2 model files to {hub_model_id}...")
                api.upload_folder(
                    folder_path=ct2_output_dir,
                    repo_id=hub_model_id,
                    repo_type="model",
                    path_in_repo="ct2_model", # Place in a subdirectory on the Hub
                    token=token
                )
                print(f"CT2 model files pushed to {hub_model_id}/ct2_model")

        except Exception as e:
            print(f"Error pushing to Hub: {e}")

    print("\n--- Training Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model for pronunciation assessment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Function to add boolean args with --no- prefix
    def add_boolean_argument(parser, name, default=False, help_true="", help_false=""):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(f'--{name}', dest=name, action='store_true', help=help_true)
        group.add_argument(f'--no-{name}', dest=name, action='store_false', help=help_false)
        parser.set_defaults(**{name: default})

    # Model Args
    parser.add_argument("--model_id", type=str, default="openai/whisper-tiny", help="Whisper model ID from Hub.")
    add_boolean_argument(parser, 'use_peft', default=True, help_true="Use PEFT/LoRA for parameter-efficient fine-tuning.", help_false="Perform full fine-tuning.")
    parser.add_argument("--peft_r", type=int, default=32, help="LoRA r value.")
    parser.add_argument("--peft_alpha", type=int, default=64, help="LoRA alpha value.")
    parser.add_argument("--peft_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--peft_target_modules", nargs='+', default=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"], help="Modules to target with LoRA.")

    # Data Args
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (Hub) or path to local CSV file.")
    parser.add_argument("--language", type=str, required=True, help="Target language short code (e.g., 'en', 'fr').")
    parser.add_argument("--language_full", type=str, required=True, help="Target language full name (e.g., 'english', 'french').")
    parser.add_argument("--train_split", type=str, default="train", help="Name of the training split.")
    parser.add_argument("--eval_split", type=str, default="eval", help="Name of the evaluation split (used if dataset_name is Hub ID or local data has eval split).")
    parser.add_argument("--text_column_name", type=str, default="sentence", help="Column name for transcripts.")
    parser.add_argument("--audio_column_name", type=str, default="audio", help="Column name for audio data/paths.")
    parser.add_argument("--max_input_length", type=float, default=30.0, help="Maximum audio duration in seconds.")
    parser.add_argument("--min_input_length", type=float, default=0.0, help="Minimum audio duration in seconds.")

    # Training Args
    parser.add_argument("--output_dir", type=str, default="./whisper-pronunciation-finetuned-test", help="Output directory.")
    add_boolean_argument(parser, 'overwrite_output_dir', default=False, help_true="Overwrite the content of the output directory.", help_false="Do not overwrite output directory.")
    add_boolean_argument(parser, 'do_train', default=True, help_true="Whether to run training.", help_false="Skip training.")
    add_boolean_argument(parser, 'do_eval', default=True, help_true="Whether to run eval on the dev set.", help_false="Skip evaluation.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=2, help="Warmup steps.")
    parser.add_argument("--max_steps", type=int, default=10, help="Total training steps.")
    add_boolean_argument(parser, 'gradient_checkpointing', default=False, help_true="Use gradient checkpointing to save memory.", help_false="Do not use gradient checkpointing.")
    add_boolean_argument(parser, 'fp16', default=torch.cuda.is_available(), help_true="Use FP16 mixed precision (GPU required).", help_false="Do not use FP16.")
    add_boolean_argument(parser, 'bf16', default=False, help_true="Use BF16 mixed precision (Ampere+ GPU required).", help_false="Do not use BF16.")
    parser.add_argument("--evaluation_strategy", dest="evaluation_strategy", type=str, default="steps", help="Evaluation strategy ('steps' or 'epoch') [DEPRECATED: use --eval_strategy]")
    parser.add_argument("--eval_strategy", dest="evaluation_strategy", type=str, default="steps", help="Evaluation strategy ('steps' or 'epoch').")
    parser.add_argument("--eval_steps", type=int, default=5, help="Evaluate every N steps.")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy ('steps' or 'epoch').")
    parser.add_argument("--save_steps", type=int, default=5, help="Save checkpoint every N steps.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every N steps.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report results to ('tensorboard', 'wandb', etc.). Can be comma-separated.")
    add_boolean_argument(parser, 'load_best_model_at_end', default=True, help_true="Load the best model at the end of training.", help_false="Do not load the best model at the end.")
    parser.add_argument("--metric_for_best_model", type=str, default="wer", help="Metric to use for best model.")
    parser.add_argument("--greater_is_better", action="store_false", default=False, help="Whether the metric is better when lower (e.g., WER).")
    add_boolean_argument(parser, 'predict_with_generate', default=True, help_true="Use generate for evaluation (needed for WER).", help_false="Do not use generate for evaluation.")
    parser.add_argument("--generation_max_length", type=int, default=225, help="Max generation length.")
    parser.add_argument("--dataloader_num_workers", type=int, default=1, help="Number of workers for dataloader.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Post-Training Args
    add_boolean_argument(parser, 'do_ct2_conversion', default=False, help_true="Convert the final model to CTranslate2 format.", help_false="Skip CTranslate2 conversion.")
    parser.add_argument("--ct2_quantization", type=str, default="float16", help="CTranslate2 quantization (e.g., 'float16', 'int8', 'int8_float16').")
    add_boolean_argument(parser, 'push_to_hub', default=False, help_true="Push the final model to Hugging Face Hub.", help_false="Do not push to Hub.")
    parser.add_argument("--final_model_name", type=str, default=None, help="Repository name for the final model on the Hub (required for push).")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token (if not logged in).")


    parsed_args = parser.parse_args()

    # Validate args
    if parsed_args.bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is specified but not supported by the current setup.")
    if parsed_args.bf16 and parsed_args.fp16:
         print("Warning: Both BF16 and FP16 are set. Disabling FP16.")
         parsed_args.fp16 = False
    if not parsed_args.do_train and not parsed_args.do_eval:
        raise ValueError("At least one of --do_train or --do_eval must be True.")
    # Updated validation for push_to_hub
    if parsed_args.push_to_hub and not parsed_args.final_model_name:
        raise ValueError("--final_model_name must be specified when --push_to_hub is True.")
    # Check if output dir should be overwritten only if training is intended
    if parsed_args.do_train and os.path.exists(parsed_args.output_dir) and os.listdir(parsed_args.output_dir) and not parsed_args.overwrite_output_dir:
        raise ValueError(f"Output directory ({parsed_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")


    main(parsed_args) 