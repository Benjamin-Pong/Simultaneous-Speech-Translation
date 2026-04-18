from types import SimpleNamespace
from typing import List

import numpy as np
from transformers import AutoProcessor, SeamlessM4TModel, SeamlessM4Tv2Model
from simulstream.server.speech_processors.incremental_output import IncrementalOutput 
from .model_sm4t import DynamicSeamlessM4T
import torch
from peft import PeftModel
from huggingface_hub import hf_hub_download


from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.seamless_sliding_window_retranslation import \
    SeamlessSlidingWindowRetranslator

class DynamicSlidingWindow(SeamlessSlidingWindowRetranslator):
    '''
    Runs stream att policy on custom model
    '''
    @classmethod
    def load_model(cls, config: SimpleNamespace):
        # only load base model here — LoRA loaded later in set_target_language
        if not hasattr(cls, "model") or cls.model is None:
            cls.processor = AutoProcessor.from_pretrained(config.hf_model_name)
            cls.model = DynamicSeamlessM4T.from_pretrained(config.hf_model_name)
            original_conformer = SeamlessM4TModel.from_pretrained(
                "facebook/hf-seamless-m4t-medium"
            ).speech_encoder.encoder.state_dict()
            cls.model.speech_encoder.encoder.load_state_dict(original_conformer, strict=False)
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls.model.to(cls.device)

    def set_target_language(self, language: str) -> None:
        # call parent to set self.tgt_lang_tag
        super().set_target_language(language)

        # load LoRA and scheduler based on language — only once
        if not hasattr(self.__class__, '_lora_loaded'):
            COMMIT_ID = "main"
            if language == "cmn":
                model_hub = "Benjaminpwh/sst-en_zh_5_sm4t_medium_lora"
                COMMIT_ID = "15dd61f3dde13976f40e3f787b56cccec94ee1b9"
                print("tgt lang: chinese")
                
            elif language == "deu":
                model_hub = "Benjaminpwh/sst-en_de_5_sm4t_medium_lora"
                print("tgt lang: german")
            elif language == "ita":
                model_hub = "Benjaminpwh/sst-en_it_6_sm4t_medium_lora"
                print("tgt lang: italian")
                COMMIT_ID = "9ffc7d7e1e345de1df0fb003b5b35af171ccbbfc"
            else:
                raise ValueError(f"Unsupported language: {language}")

            extra_weights_path = hf_hub_download(
                repo_id=model_hub,
                filename="scheduler_weights.pt"
            )
            extra_weights = torch.load(extra_weights_path, map_location="cpu")
            stripped_weights = {
                k.replace("base_model.model.", ""): v
                for k, v in extra_weights.items()
            }
            missing, unexpected = self.__class__.model.load_state_dict(
                stripped_weights, strict=False)
            print(f"Scheduler weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")

            self.__class__.model = PeftModel.from_pretrained(
            self.__class__.model, model_hub, revision=COMMIT_ID)
            self.__class__.model = self.__class__.model.merge_and_unload()
            self.__class__._lora_loaded = True
    '''
    def _build_incremental_outputs(self, generated_tokens: List[str]) -> IncrementalOutput:
        # Debug prints before calling parent
        print(f"text_history length: {len(self.text_history) if self.text_history else 0}")
        print(f"text_history: {self.text_history}")
        print(f"generated_tokens length: {len(generated_tokens)}")
        print(f"generated_tokens: {generated_tokens}")
        
        if self.text_history and len(self.text_history) > 0:
            from difflib import SequenceMatcher
            seq_matcher = SequenceMatcher(
                None, self.text_history, generated_tokens, autojunk=False)
            longest_match = seq_matcher.find_longest_match()
            print(f"longest_match.size: {longest_match.size}")
            print(f"threshold: {self.matching_threshold * len(generated_tokens)}")
            print(f"match passes: {longest_match.size >= self.matching_threshold * len(generated_tokens)}")
        
        print("---")
        
        return super()._build_incremental_outputs(generated_tokens)
    def process_chunk(self, waveform: np.float32):
        speech = self._preprocess(waveform)
        generated_tokens = self._generate(speech)
        generated_tokens = self._remove_repetitions(generated_tokens)
        return self._build_incremental_outputs(generated_tokens)

    def _remove_repetitions(self, tokens, max_repeat=3):
        """
        Truncate tokens at the point where a ngram repeats more than max_repeat times.
        """
        for ngram_size in [1, 2, 3]:
            for i in range(len(tokens) - ngram_size * max_repeat):
                ngram = tuple(tokens[i:i + ngram_size])
                # count consecutive repeats from position i
                count = 0
                j = i
                while j + ngram_size <= len(tokens) and \
                      tuple(tokens[j:j + ngram_size]) == ngram:
                    count += 1
                    j += ngram_size
                if count > max_repeat:
                    return tokens[:i]  # truncate at first repetition
        return tokens
    '''