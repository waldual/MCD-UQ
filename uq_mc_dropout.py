import jiwer
import numpy as np
import matplotlib.pyplot as plt
import Levenshtein

class uq_MC_dropout:

    def __init__(self, tamano_grupo):
        self.tamano_grupo = tamano_grupo

    def compute_wers(self, transcriptions_all, gt_texts):
        wers = []
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

        extended_gt_texts = []
        for text in gt_texts:
            extended_gt_texts.extend([text] * self.tamano_grupo)

        for i in range(len(transcriptions_all)):
            gt_text = extended_gt_texts[i]
            transcription_text = transcriptions_all[i]
            print("Processing gt_text ", gt_text)
            print("Processing transcription_text ", transcription_text)
            # Compute a per sentence word error rate
            wer = jiwer.wer(
                [gt_text],
                [transcription_text],
                truth_transform=transforms,
                hypothesis_transform=transforms,
            )
            wers.append(wer)
            print(f"Word Error Rate (WER) :", wer, "of audio ", i)
        return wers

    def compute_ece(self, certainties, wers):
        certainties_np = np.array(certainties)
        wers_np = np.array(wers)
        r = np.corrcoef(certainties_np, wers_np)
        plt.scatter(certainties_np, wers)
        plt.xlabel('Uncertainty')
        plt.ylabel('WER')
        plt.title('WER vs Uncertainty')
        plt.show()
        return r[0, 1]

    def calcular_distancias_levensthein(self, oracion, transcripciones):
        distancias = []
        longitudes_maximas = []

        for t in transcripciones:
            distancia = Levenshtein.distance(oracion, t)
            distancias.append(distancia)
            longitud_maxima = max(len(oracion), len(t))
            longitudes_maximas.append(longitud_maxima)

        return distancias, longitudes_maximas

    def encontrar_medoid(self, transcripciones):
        medoid = None
        min_total_distance = float('inf')

        for t1 in transcripciones:
            total_distance = 0
            for t2 in transcripciones:
                total_distance += Levenshtein.distance(t1, t2)

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                medoid = t1

        return medoid, min_total_distance

    def uncertainty_MCD(self, transcripciones):
        distancias_totales = []
        medoides = []
        max = []
        for i in range(0, len(transcripciones), self.tamano_grupo):
            transcripciones_grupo = transcripciones[i:i+self.tamano_grupo-1]
            medoide, distancia_minima = self.encontrar_medoid(transcripciones_grupo)
            distancias_grupo, longitudes_maximas = self.calcular_distancias_levensthein(medoide, transcripciones_grupo)
            max.append(longitudes_maximas)
            distancias_totales.append(distancias_grupo)
            medoides.append(medoide)
            longitudes = [len(medoide) for medoide in medoides]

        return distancias_totales, medoides, max, longitudes

    def dividir_distancias(self, distancias_totales, distancias_maximas):
        resultados = []
        promedios = []

        for i in range(len(distancias_totales)):
            distancia_total = distancias_totales[i]
            distancia_maxima = distancias_maximas[i]

            for j in range(len(distancia_total)):
                resultado = distancia_total[j] / distancia_maxima[j]
                resultados.append(resultado)

        for i in range(0, len(resultados), self.tamano_grupo - 1):
            grupo = resultados[i:i + self.tamano_grupo - 1]
            promedio_grupo = sum(grupo) / len(grupo)
            promedios.append(promedio_grupo)

        return promedios
