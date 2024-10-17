import jiwer
import torch
import torch
import numpy as np
from jiwer import wer
import seaborn as sns
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor,WhisperConfig
from datasets import load_dataset, load_metric, Audio
from IPython.display import Audio as AudioDis
import librosa.display
import matplotlib.pyplot as plt
from text_unidecode import unidecode
import torch
from datasets import load_dataset
from datasets import load_dataset, load_metric, Audio
from IPython.display import Audio as AudioDis
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperConfig
#import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor,WhisperConfig

# Diccionario de palabras sin tilde a palabras con tilde

class Whisper_Evaluator():

  def __init__(self, model = None, processor = None, sampling_rate = 16000):
    """
    Builds an instance of Whisper Evaluator
    param model: whisper model to evaluate
    param processor: whisper audio to feature representation (MCC based) converter
    param sampling_rate: use 16 Khz by default
    return instance of the class
    """
    self.sampling_rate = sampling_rate
    self.model = model
    self.processor = processor
    self.diccionario_tildes = {
        "proximas": "próximas",
        "tambien": "también",
        "pasaria": "pasaría",
        "raton": "ratón",
        "mas":"más",
        "autentico":"auténtico",
        "termino":"términos",
        "publico":"público",
        "si":"sí",
        "tu":"tú",
        "relacion":"relación",
        "dificil":"difícil",
        "biologica":"biológica",
        "comun":"común"
    }

  def tildar_oracion(self, oracion):
      """
      Puts accents to the spanish words needed
      param oracion: spanish sentence
      return oracion with accents
      """
      palabras = oracion.split()
      palabras_tildadas = [self.diccionario_tildes.get(palabra, palabra) for palabra in palabras]
      oracion_tildada = " ".join(palabras_tildadas)
      return oracion_tildada

  def transcribe_audio(self, audio_array):
    """Transform the audio to the input using the required representations
    param audio_array: audio array
    param model: model
    param processor: processor
    param sampling_rate: sampling rate
    return transcription 
    """
    inputs = self.processor.feature_extractor(audio_array, return_tensors="pt", sampling_rate = self.sampling_rate).input_features
    inputs = inputs.to(self.model.device)
    #print(type(inputs))
    #output decoder
    forced_decoder_ids = self.processor.get_decoder_prompt_ids(language = "es", task = "transcribe")
    with torch.no_grad():
        self.model.train()
        generated_ids = self.model.generate(
            inputs,
            forced_decoder_ids=forced_decoder_ids,
            num_return_sequences=1
        )
        #calculate logits of the model, better use logits
        logits = self.model(inputs, decoder_input_ids=generated_ids).logits
        #fetch transcriptions in text
        transcriptions = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens = False, normalize = True)
        #adds accents
        transcription = self.tildar_oracion(transcriptions[0])
    return transcription

  def transcribe_dataset(self, dataset_audios):
    """
    Transcribes an entire dataset with the format audio["audio"]["array"], audio["audio"]["sampling_rate"]
    and audio["sentence"]
    param dataset_audios: dataset of audios
    param model: model
    param processor: processor
    return transcriptions_list and the corresponding groundtruth 
    """
    #make sure the sampling rate is always 16 kHz
    dataset_audios = dataset_audios.cast_column("audio", Audio(sampling_rate= self.sampling_rate))
    transcriptions_list = []
    gt_list = []
    total_num_audios = len(dataset_audios)
    i = 0
    for audio in dataset_audios:
      print("Transcribing and uq of audio ", i, " of ", total_num_audios, end = "\r")
      i += 1
      #fetches the audio array
      audio_array = audio["audio"]["array"]
      gt_sentence = audio["sentence"]
      gt_list.append(gt_sentence)
      sampling_rate = audio["audio"]["sampling_rate"]
      #transcribe the audio
      transcription = self.transcribe_audio(audio_array)    
      transcriptions_list.append(transcription)   
    
    return transcriptions_list, gt_list

  def compute_wers(self, transcriptions_all, gt_texts):
    """
    Compute the word error rate per audio 
    param transcriptions_all: list of transcriptions
    param gt_texts: list of groundtruth texts
    return wers in a list
    """
    wers = []
    #preprocessing before computing the WER
    transforms = jiwer.Compose(
        [
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords()

        ]
    )
    #go through each sentence
    for i in range(len(transcriptions_all)):
      gt_text = gt_texts[i]
      transcription_text = transcriptions_all[i]
    
      #compute a per sentence word error rate
      wer = jiwer.wer(
                      [gt_text],
                      [transcription_text],
                      truth_transform=transforms,
                      hypothesis_transform=transforms,
                  )
      wers.append(wer)
      #print(f"Word Error Rate (WER) :", wer, "of audio ", i)
    return wers
  
  def transcriptions_MCD(self, audio_array, num_transcriptions):
        certainties_list = []
        transcriptions_list = []

        for _ in range(num_transcriptions):
            inputs = self.processor.feature_extractor(audio_array, return_tensors="pt", sampling_rate=self.sampling_rate).input_features
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="es", task="transcribe")

            with torch.no_grad():
                self.model.train()
                generated_ids = self.model.generate(
                    inputs,
                    forced_decoder_ids=forced_decoder_ids,
                    num_return_sequences=1
                )

                logits = self.model(inputs, decoder_input_ids=generated_ids).logits
                probabilities = torch.softmax(logits, dim=-1)
                max_probabilities = torch.max(probabilities, dim=2).values
                certainty = max_probabilities.mean()

                transcriptions = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False, normalize=True)
                transcription = transcriptions[0]

            certainties_list.append(certainty.item())
            transcriptions_list.append(transcription)

        return certainties_list, transcriptions_list

  def transcriptions_MCD_dataset(self, dataset_audios, num_transcriptions=1):
        certainties_list = []
        transcriptions_list = []
        total_num_audios = len(dataset_audios)

        for i, audio in enumerate(dataset_audios):
            print("Transcribing audio", i + 1, "of", total_num_audios)
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]

            certainties, transcriptions = self.transcriptions_MCD(audio_array, num_transcriptions)

            certainties_list.extend(certainties)
            transcriptions_list.extend(transcriptions)

        return certainties_list, transcriptions_list