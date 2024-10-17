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
#use CUDA gpu if available
from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import snapshot_download
import pandas as pd
import librosa
from zipfile import ZipFile

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Dataset_loader():
    def __init__(self):
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

    def load_dataset_ciempies(self):
      """
      Load dataset from ciempiess dataset
      """
      dataset = load_dataset("ciempiess/ciempiess_test", split="test")
      dataset = dataset.cast_column("audio", Audio(sampling_rate = 16_000))
      #fetch audios
      audios = load_dataset("ciempiess/ciempiess_test")["test"]["audio"]
      #fetch texts from groundtruth
      texts = dataset["normalized_text"] #transcripciones
      #print("texts ", texts)
      return audios, texts
      
      
    def load_dataset_raw_uq(repo_id="saul1917/SpaTrans_UQ_Bench_raw",partition_type = "test", partition = 10, audio_extension = ".wav"):
      """""
      Load dataset from hugging face repo
      param repo_id: id of the repo
      param partition_type: type of the partition (test, calibration or fine_tune)
      param partition: number of the partition (0 to 10)
      return audios and texts
      """
      #dataset template
      dictionary_dataset = {}
      #dataset from hugging face
      #dataset = Dataset.from_dict(dict_dataset)
      #list of transcriptions
      transcriptions = []
      #list of dictionaries with audio file and metadata
      list_dict_audios = []
      #create the partition id
      partition_id = "partition_" + str(partition)
      path_downloaded = snapshot_download(repo_id="saul1917/SpaTrans_UQ_Bench_raw", repo_type="dataset")
      print("path_downloaded ", path_downloaded)
      #create the full in and out path
      full_path_audios_zip = path_downloaded + "/" + partition_id + "/" + partition_type + ".zip"
      full_path_csv =  path_downloaded + "/" + partition_id + "/" + partition_type + ".csv"
      full_path_audios_out_unzip = path_downloaded + "/" + partition_id + "/"
      full_path_audios_out_target = path_downloaded + "/" + partition_id + "/" + partition_type + "/"
      #read the metadata
      data_frame_metadata = pd.read_csv(full_path_csv)
      #extract the files in the full path corresponding to the audios
      try:
        rar_file = ZipFile(full_path_audios_zip, 'r')
        rar_file.extractall(path = full_path_audios_out_unzip)
        rar_file.close()
      except:
        print("Error extracting audios, could be that they were already extracted")

      #go through each record
      for index, row in data_frame_metadata.iterrows():
        dictionary_entry = {}
        dictionary_audio = {}
        audio_id = row["audio_id"]
        #all audios are stored in wav
        full_audio_name = full_path_audios_out_target + audio_id + audio_extension
        #open audio file
        #print("trying to read audio ", full_audio_name)
        audio, sample_rate = librosa.load(full_audio_name)
        #create the dictionary with audio data
        dictionary_audio["array"] = audio
        dictionary_audio["sampling_rate"] = sample_rate
        transcriptions.append(row["transcription"])

        dictionary_entry["sentence"] = row["transcription"]
        dictionary_entry["audio"] = dictionary_audio
        #append the dictionary with audio data
        list_dict_audios.append(dictionary_entry)
      #cast it to Dataset from huggingface
      dataset_hf = Dataset.from_list(list_dict_audios)
      #print("list_dict_audios ", list_dict_audios)
      return dataset_hf
      
    def contaminate_audio_array(self, audio_array, noise_audios_dataset, weight_noise = 0.6, sampling_rate = 16000):
        """""
        Contaminate audio with noise
        param weight_noise: weight of noise in the audio
        param sampling_rate: sampling rate of the audio
        param audio_array: input audio array
        param noise_audios_dataset: noise dataset
        return audio_combined_array
        """
        sample_clean_data_array = audio_array

        #load noisy dataset

        train_dataset = noise_audios_dataset['train']
        noise_audios = train_dataset['audio']
        sample_noise = noise_audios[1]
        sample_noise_array = sample_noise["array"]

        #Make sample_noise_array audio have the same length of sample_clean_data_array
        min_length =len(sample_clean_data_array)
        if len(sample_noise_array)<min_length:
          sample_ruido_data_array=np.tile(sample_noise_array, int(np.round(min_length/len(sample_noise_array))+1))[:min_length]
        else:
          sample_ruido_data_array = sample_noise_array[:min_length]
        #weight clean and noisy audio contribution
        weight_clean =  1 - weight_noise
        audio_combined_array = (weight_clean * sample_clean_data_array) + (weight_noise * sample_ruido_data_array)
        # Debugging
        AudioDis(data = audio_combined_array, rate = sampling_rate)
        return audio_combined_array
      
    def tildar_oracion(self, oracion):
        """
        Adds spanish accents
        param oracion: sentence to add spanish accents
        returns the sentences
        """
        palabras = oracion.split()
        palabras_tildadas = [self.diccionario_tildes.get(palabra, palabra) for palabra in palabras]
        oracion_tildada = " ".join(palabras_tildadas)
        return oracion_tildada