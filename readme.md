# Projekt 3

## 1. Raport: Generowanie Obrazów (Neural Renderer) - Phong shading*

Celem projektu było stworzenie modelu generatywnego zdolnego do odtwarzania obiektów 3D (odtworzenie Phong Shading) na podstawie zadanych parametrów. Poniżej przedstawiono ewolucję trzech modeli, napotkane problemy oraz finalne wyniki.

[Link do folderu z wynikami z modelu 1 i notatnikami](https://drive.google.com/drive/folders/1wM5_MUKcisrMO29oBGaAhzGDBwIgoPq7)

[Link do folderu z wynikami z pozostałych modeli i kodem w plikach w formacie PY](https://drive.google.com/drive/u/0/folders/15SehDe59HaoV9Bu6N-18cYAcZ1BwxG56)

## 2. Przegląd Modeli

### Model 0: Podejście Bazowe ("Upsample")
* **Architektura:** Prosta sieć oparta na operacji Upsample (skalowanie).
* **Charakterystyka:** Model bazowy, który posłużył jako punkt odniesienia. Miał trudności z generowaniem ostrych detali i stabilnością treningu.
* **Dataset:** Zbiór domyślny (małe obiekty, parametry z zadania - z jedynie lekkim ograniczeniem na częstość występowania całościowo czarnych obrazków).
* Model nie generował wystarczająco dobrych obrazów

### Model 1: Wersja modelu 2 bez warstwy normalizującej 1D i na innym datasetcie
* **Architektura i charakterystyka:** Jest on podobny do modelu 2, który używał dyskryminatora z mniejszą liczbę kanałów w warstwach konwolucyjnych 2D i liniowych oraz mniejszą ilością tych warstw w dyskryminatorze. Natomiast ten model za to nie zawiera w architekurze warstwy normalizującej na dane o 1 wymiarze.
* **Dataset:** Zbiór według parametrów domyślnych kodu otrzymanego wraz z opisem projektu.
* Checkpoint Model 1: checkpoints_v6/gen_epoch_400.pth  -> epoka 400

### Model 2: Architektura Deconvolution ("ConvTranspose2d")
* **Architektura:** Wprowadzenie warstw transponowanej konwolucji.
* **Dataset:** Zmodyfikowany zbiór "łatwy" (większe obiekty, wyeliminowanie pustych obrazów).
* checkpoint: checkpoints_3/gen_epoch_289.pth -> epoka 289


### Model 3: Architektura hybrydowa ("Upsample + Conv")
* **Architektura:** Ulepszona wersja, głębsza niż Model 2. Zastąpiono dekonwolucję podejściem "Upsample + Convolution". Pozwala to na gładsze generowanie obrazu i eliminuje artefakty typowe dla Modelu 2.
* **Dataset:** Zbiór "łatwy" (duże obiekty).
* checkpoint: checkpoints_3/gen_epoch_270.pth -> epoka 270

## 3. Przebieg eksperymentów 

### A. Trening modelu 1

* **Strategia treningu:** Złożony proces wieloetapowy:
    1.  **Warmup:** Wstępne uczenie samego generatora na funkcji straty Masked-L1.
    2.  **Trening Adaptacyjny:** Dynamiczna pętla treningowa ("adaptacyjny" trening), która monitoruje siłę Dyskryminatora i Generatora, "boostując" (dodatkowo trenując) stronę, która zaczyna przegrywać.
    3.  **Fine-tuning:** Precyzyjne manualne dobieranie współczynnika uczenia (Learning Rate) oraz wagi straty Masked-L1 w celu wyjścia z minimów lokalnych.
* **Wynik najleszy L1 Loss (raw_l1):** około 0.1

Parametry treningowe tego modelu zostały zaprezentowane w tabeli 1, a według wykresu (rys. 1) trening był niestabilny, bo strata generatora i dyskryminator rosła. Ogólnie dyskryminator rozpoznawał fałszywe obrazy tylko, że w kolejnych epokach wycodziło mu to gorzej.

 | LEARNING_RATE_GEN | LEARNING_RATE_DISC | BATCH_SIZE | Liczba epok | L1_LAMBDA |
 | :--- | :--- | :--- | :--- | :--- |
 | 0.0002 | 0.00002 | 16 | 400 | 30|

*Tabela 1: Parametry treningowe na modelu 2*

![plot_model_6](/img/plot_6.svg)
    *Rys. 1: Wykresy procesu uczenia modelu 2.*

### B. Trening modelu 2

* **Strategia treningu:** Podobna do modelu 1.
* **Wynik (Średni L1 Loss):** 0.042313

Oprócz najlepszej wersji na model 2 było jeszcze 5 kandydatów, którzy różnili się jedynie pod względem parametrów treningowych pokazanych w tabeli 2 oraz tym, że byli oni trenowani na zbiorze zgodnym według domyślnych wartości kodu udostępnionego wraz z treścią projektu.
Według wykresów (rys. 2-6) dyskryminator ogólnie odróżniał fałszywe obrazy w tym, że dla wersji innej niż 1 lub 3 wychodził mu to gorzej. Strata dla generatora spadała głównie jedynie na początku, gdy strata dyskryminatora fluktuowała przeważnie. Jednak od 4 wersji zaczęła ona rosnąć.

| Wersja | LEARNING_RATE_GEN | LEARNING_RATE_DISC | BATCH_SIZE | Liczba epok | L1_LAMBDA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 0.0002 | 0.00002 | 64 | 100 | 30|
| **2** | 0.00002 | 0.000002 | 16 | 100 | 10|
| **3** | 0.00002 | 0.00002 | 16 | 300 | 25|
| **4** | 0.0002 | 0.00002 | 16 | 250 | 30|
| **5** | 0.0001 | 0.00001 | 16 | 400 | 35|

*Tabela 2: Parametry treningowe na modelu 1*

![plot_model_1](/img/plot_1.svg)
    *Rys. 2: Wykresy procesu uczenia dla wersji 1.*

![plot_model_2](/img/plot_2.svg)
    *Rys. 3: Wykresy procesu uczenia dla wersji 2.*

![plot_model_3](/img/plot_3.svg)
    *Rys. 4: Wykresy procesu uczenia dla wersji 3.*

![plot_model_4](/img/plot_4.svg)
    *Rys. 5: Wykresy procesu uczenia dla wersji 4.*

![plot_model_5](/img/plot_5.svg)
    *Rys. 6: Wykresy procesu uczenia dla wersji 5.*

### C. Trening modelu 3

* **Strategia treningu:** Podobna do modelu 1, z silnym naciskiem na trening adaptacyjny. Rozwiązano tu kluczowy problem zbyt silnego dyskryminatora poprzez modyfikację tempa uczenia (LR) generatora względem dyskryminatora.
* **Wynik (Średni L1 Loss):** 0.037179 (Najlepszy wynik precyzji geometrycznej).

## 4. Wyzwania

W trakcie pracy zidentyfikowano trzy główne obszary problemowe, które wymagały niestandardowych rozwiązań.

### A. Walka z Dyskryminatorem (Balans G i D)
Największym wyzwaniem w treningu GAN było utrzymanie równowagi między Generatorem a Dyskryminatorem.

* **Wyzwanie 1: "Zbyt silny krytyk":** W początkowych fazach Dyskryminator uczył się za szybko, odrzucając wszystkie próby Generatora jako fałsz (Accuracy bliskie 100%). Generator nie otrzymywał wtedy użytecznego gradientu (kierunku poprawy), co prowadziło do stagnacji. Rozwiązano to poprzez uproszczenie architektury Dyskryminatora (na nadal w całości nie wystarczyło stąd rozwiązanie nr 2).
* **Wyzwanie 2: Overfitting:** Dyskryminator zamiast uczyć się cech obiektu i generalizować, zapamiętywał obrazy treningowe na pamięć.
* **Zastosowane Rozwiązania:**
    * **Trening Adaptacyjny:** Zaimplementowano pętlę logiczną monitorującą skuteczność (Accuracy średnie, dla fake i dla real obrazów). Jeśli Dyskryminator dominował -> wstrzymywano jego trening i trenowano tylko Generator (nawet dwukrotnie na tym samym batchu, aby wzmocnić generator w stosunku do Dyksryminatora). Oraz odwrotnie gdy Dyskryminator był za słaby (praktycznie się nie zdarzało w modelu 3, w przeciwnieństwie do modelu 2).
    * **Regularyzacja:** Wprowadzono **Dropout** (losowe wyłączanie neuronów) oraz **Noise Injection** (szum na wejściu), aby utrudnić Dyskryminatorowi zapamiętywanie, zamiast generalizowania.

### B. Dataset i trudność zadania
W parametrach przekazywanych do modelu zastosowano odpowiednie normalizacje, oraz relatywne wartości parametrów dotyczących położenia - kamera-obiekt-światło.

* **Wyzwanie:** Pierwotny zestaw danych (zbiór "domyślny" - na podanych parametrach z opisu projektu + dodatkowym warunkom logicznym i lekkiemu ograniczeniu pozycji tak aby było mniej w całości czarnych obrazów) zawierał bardzo małe obiekty (zajmujące <5% obrazu) lub puste obrazy. Sieć miała tendencję do generowania czystego czarnego tła, ponieważ statystycznie dawało to mały błąd globalny, ignorując trudny w rysowaniu obiekt.
* **Rozwiązanie:** Stworzono dataset na innych bardziej ograniczonych parametrach, gdzie obiekty stanowią min. 20% powierzchni (oraz prawie bez pustych obrazów). Pozwoliło to sieci najpierw nauczyć się generować poprawne kształty, co w założeniu miało stanowić bazę do późniejszego fine-tuningu na trudniejszych danych (najpierw trenowanie na zbiorze łatwym potem na domyślnym). Ze względu na brak czasu i dalej nie do końca zadowalające wyniki (które dawałyby perspektywy na poradzenie sobie z pierwotnym datasetem) niedokonano fine-tuningu.

### C. Funkcje straty
* **L1:** Początkowo stosowano zwykłą stratę L1, ale to nie wymuszało precyzji na obiekcie (tym bardziej na domyślnym zbiorze).
* **Masked L1:** Wprowadzono maskowanie – błąd na pikselach należących do obiektu był karany znacznie surowiej (wysoki współczynnik wagi) niż błąd na tle. Zmusiło to generator do "skupienia się" na obiekcie.
* **VGG:** Eksperymenty ze stratą percepcyjną (VGG Loss) wydłużyły znacznie czas treningu, nie przynosząc zauważalnej poprawy jakości wizualnej w tym zadaniu. Odrzucono tę funkcję.

## 5. Wyniki

### Metryki

Poniższe tabele prezentują wyniki dla zbioru testowego (600 obrazów).


| Metoda | FLIP ($\downarrow$) | LPIPS ($\downarrow$) | SSIM ($\uparrow$) | Hausdorff ($\downarrow$) |
| :--- | :--- | :--- | :--- | :--- |
| **neural_renderer_1** | 0.347 | 0.243 | 0.104 | 102.528 |
| **neural_renderer_2** | 0.062 | 0.162 | 0.637 | 55.869 |
| **neural_renderer_3** | **0.057** | **0.156** | **0.728** | **43.411** |

*Tabela 3: Wyniki metryk z modeli 1, 2 i 3*

W modelach widocznych w tabeli 3 zastosowano dwie różne architektury, dwa sposoby treningu oraz 2 datasety - na domyślnych parametrach generatora do renderowania sceny Phonga otrzymanego wraz z treścią zadania (model 1) i uproszczony (modele 2 i 3) w stosunku do zadanych parametrów z opisu projektu.  

Z niej widać wyraźny skok jakościowy między modelem 1 a 2, oraz finalne doszlifowanie wyników w modelu 3 (ulepszona architektura i od początku lepsze podejście podczas treningu), szczególnie w metryce strukturalnej (SSIM) i odległości Hausdorffa. Mimo innych właściwości datasetu dla modelu 1, które były otrzymane za pomocą wcześniej wspomnianego generatora, który tworzy prostszy zbiór na podstawie przeanalizowanego kodu, nie osiągnał porównywalnych wyników z modelami 2 i 3. 

### Wizualizacje

Poniżej zaprezentowano wyniki wizualne wygenerowane z modeli 1, 2 i 3. Na podstawie nich model 1 generuje obrazy słabej jakości. Położenie czasem jest prawidłowe, a czasem inne. Kolory główne (bazowe) kuli są prawidłowe, ale nienaturalne ze względu na występowanie artefaktów. Natomiast model 2 i 3 stworzą kule w tym samym miejscu jak na obrazach referencyjnych, które są mniej rozpikselowane. Nie mają na sobie one w większosci dziwnych wróżniających przebarwień. Niektóre obrazy wynikowe z modelu 3, biorąc pod uwagę rys. 15 mogą wyjść trochę żywe.

#### Model 1

![wiz_model_1](/img/m_1.png)

*Rys. 7: Po lewej obraz referencyjny, a po prawej obraz wynikowy z modelu 1.*

![wiz_model_1_2](/img/m_1_2.png)

*Rys. 8: Po lewej obraz referencyjny, a po prawej obraz wynikowy z modelu 1.*

![wiz_model_1_3](/img/m_1_3.png)

*Rys. 9: Po lewej obraz referencyjny, a po prawej obraz wynikowy z modelu 1.*

#### Model 2

![wiz_model_2](/img/m_2.png)

*Rys. 10: Po lewej obraz referencyjny, a po prawej obraz wynikowy z modelu 2.*

![wiz_model_2_2](/img/m_2_2.png)

*Rys. 11: Po lewej obraz referencyjny, a po prawej obraz wynikowy z modelu 2.*

![wiz_model_2_3](/img/m_2_3.png)

*Rys. 12: Po lewej obraz referencyjny, a po prawej obraz wynikowy z modelu 2.*


#### Model 3

![wiz_model_3](/img/m_3.png)

*Rys. 13: Po lewej obraz referencyjny, a po prawej obraz wynikowy z modelu 3.*

![wiz_model_3](/img/m_3_2.png)

*Rys. 14: Po lewej obraz referencyjny, a po prawej obraz wynikowy z modelu 3.*

![wiz_model_3](/img/m_3_3.png)

*Rys. 15: Po lewej obraz referencyjny, a po prawej obraz wynikowy z modelu 3.*

## 6. Podsumowanie i wnioski

Na podstawie przeprowadzonych eksperymentów wyciągnięto następujące wnioski:

1.  Architektura hybrydowa ("Upsample + Conv") okazała się najlepsza. Jest bardziej stabilna w treningu, jest w stanie generować gładsze obrazy, unikając artefaktów geometrycznych.
2.  Kluczem do sukcesu nie była sama sieć (duża różnica w architekturze modelu i modelu 2, brak różnic architektury modelu 1 i modelu 2), ale sposób jej trenowania. Odpowiedni fine-tuning i dynamiczne balansowanie sił między Generatorem a Dyskryminatorem pozwoliło wyjść z impasu, w którym Dyskryminator blokował rozwój Generatora.
3. Model 3 osiąga najlepsze wyniki zarówno w metrykach geometrycznych (Hausdorff), jak i percepcyjnych (LPIPS/SSIM). Oznacza to, że generuje obrazy nie tylko poprawne matematycznie (kształt), ale i najbardziej przekonujące dla ludzkiego oka (cieniowanie, faktura).  
4. Wynikiem lepszych wartości w metrykach jest w dużej mierze zasługą lepszej architektury i procesu treningowemu modeli 2 i 3, ale również wpływ na to mają cechy datasetu, na którym trenuje dany model. Proces treningu może być słaby, jeśli zbiór danych jest zbyt uproszczony i za mało różnorodny. Warto, więc mieć takie dane, które są między 2 skrajnościami, a potem ewentualnie poszerzać zakres danych.