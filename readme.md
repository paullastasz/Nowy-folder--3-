# Model 1
"upsample"

![plot_model_1](/img/plot.svg)

*Rys. 1: Wykresy z .*

# Model 2
- model: model_2.py -> "prosta" architektura (v2) "ConvTranspose2d"
- checkpoint: checkpoints_2/289.pth -> dlugi trening, warmup generatora, potem "statyczny trening", dotrenowanie dyskryminatora, trening "adaptacyjny" (boost D i G zależnie od potrzeb -> statystyk D), fine-tuning poprzez dobór LR (w celu wyjścia z minimum lokalnego zwiększenie potem zmniejszenie) oraz przez dobór współczynnika straty masked-L1.
- dataset -> dataset "łatwy" - z wiekszymi obiektami + wszystkie mają obiekt
- Średni L1 Loss: 0.042313

# Model 3
- model: model_3.py -> "lepsza" architektura (v3) "Upsample + Conv", głębsze niż model 2
- checkpoint: checkpoints_3/270.pth  -> podobnie jak model 2 (z przewagą treningu "adaptacyjnego") - obecne problemy z dyskryminatorem który był za mocny, rozwiazano modyfikacja LR G
- dataset -> dataset "łatwy" - z wiekszymi obiektami + wszystkie mają obiekt
- Średni L1 Loss: 0.037179

# Eksperymenty

## Dyskryminator i trening: 

Sprawdzono dwie architektury (pierwsza zdecydowanie za mocna) i druga - prostsza.

Problemem w modelu 1 oprócz "trudniejszego" datasetu, był też ciężki dyskryminator który był za mocny w porównaniu do generatora.

Pojawiającym się później problemem z dyskryminatorem było to że uczył się na pamięć obrazków treningowych zamiast generalizować, overfitting.

Skutecznymi pomysłami było dodanie dropoutu oraz szumu w trakcie treningu gdy D był za mocny. Zależnie od potrzeb ręcznie modyfikowano dropout i szum.

Jednoczesnie problemem był za słabo uczący się dyskryminator (przy modelu 2) - gdy był za słaby (niskie stałe LR) to Generator bez problemu go oszukiwał ale nie szedł w stronę realizmu.

Jak miał wyższe LR to był za mocny, generator nie dawał rady, w praktyce nie dostawał gradientu/kierunku bo wszystko co zrobił było oceniane przez D jako całkowity fałsz.
Stąd pomysł na automatyczny trening "adaptacyjny" na podstawie Accuracy Dyskryminator (dla Real i False oraz średniej) - jeśli D był za mocny (lub leniwy, wszystko prawda) to nie trenowano go, i przez kilka batchy trenowano G aż D osłabł (lub max trenowań na batchu).
Jak D był za słaby (niskie accuracy)(lub był haterem - wszystko fałsz) to go nawet kilka razy dotrenowywano na batchu aż się poprawią statystyki, potem na tym samym batchu trenowano G i porównywano, jak trzeba było to wzmacniano G.

Dało to znacznie lepsze wyniki w połączeniu z ustawieniem odpowiednim LR G i D oraz parametrami L1 masked.

## Dataset:

Uznano, że zaproponowane parametry są zbyt trudne (obiekt to kilka pikseli), obrazek w większości pusty, duża część obrazków nie miała wcale obiektu (tu od razu na początku dano ifa i lekką zmianę parametrów żeby ograniczyć puste obrazki do max 10%).
Obiekty stanowiły po 5% lub mniej obrazka.

Do modelu 2 zmieniono podejście - korekcja datasetu na łatwiejszy - ograniczono parametry do takich aby na każdym był obiekt (nieliczne w większości były puste bo obiekt miał tylko kilka pikseli na obrazku) o wielkości stanowiącej 20% obrazka.
Następnie pomysł był taki, aby po uzyskaniu dobrego modelu zrobić fine-tuning na trudniejszym datasetcie - z oryginalnymi parametrami.

## Funkcje strat:

Na poczatku stosowano L1 (w warmupie G), L1*wsp_1 + GAN.

Jeszcze w modelu 1 zmieniono L1 na L1_masked -> nakładamy maskę na obiekt, błędne piksele na obiekcie stanowią wsp_2 razy większą karę.
Ustalono, że podejście było słuszne, rezultaty się znacznie poprawiły na oryginalnym datasetcie obiekt stanowił z 5% obrazka lub mniej, więc maska była potrzebna aby G kładł większy nacisk na poprawne rysowanie obiektu.

Eksperymentowano też z dodaniem straty VGG ale trening był za długi, a rezultaty niezadowalające.


# Wizualizacja

# Wyniki

## Metryki

Otrzymane wartości metryk zostały obliczone na podstawie zbioru testowego o wielkości 600 obrazów. 

| Metoda       | FLIP | LPIPS       | SSIM       | Hausdorff      |
|------------|------|-------------|-------------|-------------|
|  neural renderer | 0.347   | 0.243    |  0.104|  102.528|

*Tabela 1: Metryki dla modelu 1.*

| Metoda       | FLIP | LPIPS       | SSIM       | Hausdorff      |
|------------|------|-------------|-------------|-------------|
|  neural renderer | 0.062   | 0.162    |  0.637|  55.869|

*Tabela 2: Metryki dla modelu 2.*

| Metoda       | FLIP | LPIPS       | SSIM       | Hausdorff      |
|------------|------|-------------|-------------|-------------|
|  neural renderer | 0.057  | 0.156    | 0.728 | 43.411 |

*Tabela 3: Metryki dla modelu 3.*

# Wnioski

Na podstawie przeprowadzonych eksperymentów najlepsze efekty osiąga model 3. Pod względem prepecji widzenia daje on rezultaty zbliżone do referencyjnego obrazu, a pod względem struktury osiąga również dobre wyniki.  