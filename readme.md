# Raport: Generowanie Obrazów (Neural Renderer) - Phong shading

Celem projektu było stworzenie modelu generatywnego zdolnego do odtwarzania obiektów 3D (odtworzenie Phong Shading) na podstawie zadanych parametrów. Poniżej przedstawiono ewolucję trzech modeli, napotkane problemy oraz finalne wyniki.

[Folder z wynikami 1](https://drive.google.com/drive/u/0/folders/15SehDe59HaoV9Bu6N-18cYAcZ1BwxG56)

[Folder z wynikami 2 i notatnikami](https://drive.google.com/drive/folders/1wM5_MUKcisrMO29oBGaAhzGDBwIgoPq7)

# 1. Wyniki

Poniższe tabele prezentują wyniki dla zbioru testowego (600 obrazów).


| Metoda | FLIP ($\downarrow$) | LPIPS ($\downarrow$) | SSIM ($\uparrow$) | Hausdorff ($\downarrow$) |
| :--- | :--- | :--- | :--- | :--- |
| **neural_renderer_1** | 0.347 | 0.243 | 0.104 | 102.528 |
| **neural_renderer_2** | 0.062 | 0.162 | 0.637 | 55.869 |
| **neural_renderer_3** | **0.057** | **0.156** | **0.728** | **43.411** |

*Tabela zbiorcza: Widać wyraźny skok jakościowy między modelem 1 a 2, oraz finalne doszlifowanie wyników w modelu 3 (ulepszona architektura i od początku lepsze podejście podczas treningu), szczególnie w metryce strukturalnej (SSIM) i odległości Hausdorffa.*

W modelach widocznych w tabeli zbiorczej zastosowano dwie różne architektury, dwa sposoby treningu oraz uproszczony dataset w stosunku do zadanych parametrów z opisu projektu. Poniżej opisano napotkane problemy i  przebieg procesu ich rozwiązywania.  

# 2. Przegląd Modeli


### Model 0: Podejście Bazowe ("Upsample")
* **Architektura:** Prosta sieć oparta na operacji Upsample (skalowanie).
* **Charakterystyka:** Model bazowy, który posłużył jako punkt odniesienia. Miał trudności z generowaniem ostrych detali i stabilnością treningu.
* **Dataset:** Zbiór domyślny (małe obiekty, parametry z zadania - z jedynie lekkim ograniczeniem na częstość występowania całościowo czarnych obrazków).
* Model nie generował wystarczająco dobrych obrazów

### Model 1 i 2: Architektura Deconvolution ("ConvTranspose2d")
* **Architektura:** Wprowadzenie warstw transponowanej konwolucji.
* **Strategia treningu:** Złożony proces wieloetapowy:
    1.  **Warmup:** Wstępne uczenie samego generatora na funkcji straty Masked-L1.
    2.  **Trening Adaptacyjny:** Dynamiczna pętla treningowa ("adaptacyjny" trening), która monitoruje siłę Dyskryminatora i Generatora, "boostując" (dodatkowo trenując) stronę, która zaczyna przegrywać.
    3.  **Fine-tuning:** Precyzyjne manualne dobieranie współczynnika uczenia (Learning Rate) oraz wagi straty Masked-L1 w celu wyjścia z minimów lokalnych.
* **Dataset:** Zmodyfikowany zbiór "łatwy" (większe obiekty, wyeliminowanie pustych obrazów).
* **Wynik (Średni L1 Loss):** 0.042313
* **Model 1** jest dotrenowanym (do 400 epok) modelem 2.
* Checkpoint Model 2: checkpoints_3/270.pth  -> Epoka 270 


![plot_model_1](/img/plot.svg)
    *Rys. 1: Wykresy procesu uczenia dla Modelu 1.*

### Model 3: Architektura hybrydowa ("Upsample + Conv")
* **Architektura:** Ulepszona wersja, głębsza niż Model 2. Zastąpiono dekonwolucję podejściem "Upsample + Convolution". Pozwala to na gładsze generowanie obrazu i eliminuje artefakty typowe dla Modelu 2.
* **Strategia treningu:** Podobna do Modelu 2, z silnym naciskiem na trening adaptacyjny. Rozwiązano tu kluczowy problem zbyt silnego dyskryminatora poprzez modyfikację tempa uczenia (LR) generatora względem dyskryminatora.
* **Dataset:** Zbiór "łatwy" (duże obiekty).
* **Wynik (Średni L1 Loss):** 0.037179 (Najlepszy wynik precyzji geometrycznej).
* checkpoint: checkpoints_3/270.pth -> epoka 270


# 3. Przebieg eksperymentów i wyzwania

W trakcie pracy zidentyfikowano trzy główne obszary problemowe, które wymagały niestandardowych rozwiązań.

## A. Walka z Dyskryminatorem (Balans G i D)
Największym wyzwaniem w treningu GAN było utrzymanie równowagi między Generatorem a Dyskryminatorem.

* **Wyzwanie 1: "Zbyt silny krytyk":** W początkowych fazach Dyskryminator uczył się za szybko, odrzucając wszystkie próby Generatora jako fałsz (Accuracy bliskie 100%). Generator nie otrzymywał wtedy użytecznego gradientu (kierunku poprawy), co prowadziło do stagnacji. Rozwiązano to poprzez uproszczenie architektury Dyskryminatora (na nadal w całości nie wystarczyło stąd rozwiązanie nr 2).
* **Wyzwanie 2: Overfitting:** Dyskryminator zamiast uczyć się cech obiektu i generalizować, zapamiętywał obrazy treningowe na pamięć.
* **Zastosowane Rozwiązania:**
    * **Trening Adaptacyjny:** Zaimplementowano pętlę logiczną monitorującą skuteczność (Accuracy średnie, dla fake i dla real obrazów). Jeśli Dyskryminator dominował -> wstrzymywano jego trening i trenowano tylko Generator (nawet dwukrotnie na tym samym batchu, aby wzmocnić generator w stosunku do Dyksryminatora). Oraz odwrotnie gdy Dyskryminator był za słaby (praktycznie się nie zdarzało w modelu 3, w przeciwnieństwie do modelu 2).
    * **Regularyzacja:** Wprowadzono **Dropout** (losowe wyłączanie neuronów) oraz **Noise Injection** (szum na wejściu), aby utrudnić Dyskryminatorowi zapamiętywanie, zamiast generalizowania.

## B. Dataset i trudność zadania
W parametrach przekazywanych do modelu zastosowano odpowiednie normalizacje, oraz relatywne wartości parametrów dotyczących położenia - kamera-obiekt-światło.

* **Wyzwanie:** Pierwotny zestaw danych (zbiór "domyślny" - na podanych parametrach z opisu projektu + dodatkowym warunkom logicznym i lekkiemu ograniczeniu pozycji tak aby było mniej w całości czarnych obrazów) zawierał bardzo małe obiekty (zajmujące <5% obrazu) lub puste obrazy. Sieć miała tendencję do generowania czystego czarnego tła, ponieważ statystycznie dawało to mały błąd globalny, ignorując trudny w rysowaniu obiekt.
* **Rozwiązanie:** Stworzono dataset na innych bardziej ograniczonych parametrach, gdzie obiekty stanowią min. 20% powierzchni (oraz prawie bez pustych obrazów). Pozwoliło to sieci najpierw nauczyć się generować poprawne kształty, co w założeniu miało stanowić bazę do późniejszego fine-tuningu na trudniejszych danych (najpierw trenowanie na zbiorze łatwym potem na domyślnym). Ze względu na brak czasu i dalej nie do końca zadowalające wyniki (które dawałyby perspektywy na poradzenie sobie z pierwotnym datasetem) niedokonano fine-tuningu.

## C. Funkcje straty
* **L1:** Początkowo stosowano zwykłą stratę L1, ale to nie wymuszało precyzji na obiekcie (tym bardziej na domyślnym zbiorze).
* **Masked L1:** Wprowadzono maskowanie – błąd na pikselach należących do obiektu był karany znacznie surowiej (wysoki współczynnik wagi) niż błąd na tle. Zmusiło to generator do "skupienia się" na obiekcie.
* **VGG:** Eksperymenty ze stratą percepcyjną (VGG Loss) wydłużyły znacznie czas treningu, nie przynosząc zauważalnej poprawy jakości wizualnej w tym zadaniu. Odrzucono tę funkcję.

# 4. Podsumowanie i wnioski

Na podstawie przeprowadzonych eksperymentów wyciągnięto następujące wnioski:

1.  Architektura hybrydowa ("Upsample + Conv") okazała się najlepsza. Jest bardziej stabilna w treningu, jest w stanie generować gładsze obrazy, unikając artefaktów geometrycznych.
2.  Kluczem do sukcesu nie była sama sieć (duża różnica w architekturze modelu i modelu 2, brak różnic architektury modelu 1 i modelu 2), ale sposób jej trenowania. Odpowiedni fine-tuning i dynamiczne balansowanie sił między Generatorem a Dyskryminatorem pozwoliło wyjść z impasu, w którym Dyskryminator blokował rozwój Generatora.
3. Model 3 osiąga najlepsze wyniki zarówno w metrykach geometrycznych (Hausdorff), jak i percepcyjnych (LPIPS/SSIM). Oznacza to, że generuje obrazy nie tylko poprawne matematycznie (kształt), ale i najbardziej przekonujące dla ludzkiego oka (cieniowanie, faktura).
