# Detekcija anomalija u ponašanju veće skupine ljudi

Implementacija detekcije anomalija u ponašanju veće skupine ljudi inspirirana sljedećim radovima:

* Raghavendra, R. & Del Bue, Alessio & Cristani, Marco & Murino, Vittorio. (2011). Abnormal Crowd Behavior Detection by Social Force Optimization. 134-145. 10.1007/978-3-642-25446-8_15. [https://www.researchgate.net/publication/221620704_Abnormal_Crowd_Behavior_Detection_by_Social_Force_Optimization]
* Raghavendra, R. & Del Bue, Alessio & Cristani, Marco & Murino, Vittorio. (2011). Optimizing interaction force for global anomaly detection in crowded scenes. 136-143. 10.1109/ICCVW.2011.6130235.  [https://www.researchgate.net/publication/221430050_Optimizing_interaction_force_for_global_anomaly_detection_in_crowded_scenes]
* R. Mehran, A. Oyama and M. Shah, "Abnormal crowd behavior detection using social force model," 2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, 2009, pp. 935-942.
doi: 10.1109/CVPR.2009.5206641 [https://ieeexplore.ieee.org/document/5206641]

## 0. Preduvjeti za pokretanje
* Python (Verzija 3)
* OpenCV
* numpy

## 1. Opis priloženih datoteka

* **anomalydetection.py** || Sadrži izvorni kod svih metoda i algoritama korištenih u izgradnji sustava za detekciju. Sadrži main metodu koja vodi korisnika korak po korak kroz postavljanje parametara algoritma i odabir datoteka
* **testerUCSD.py** || Služi za evaluiranje algoritma nad UCSD bazi podataka. (Zahtjeva da korisnik posjeduje UCSD bazu.)
* **testerUMN** || Služi za evaluiranje algoritma nad UMN bazi podataka. (Zahtjeva da korisnik posjeduje UMN bazu podataka.)
* **demo.py** || Pokreće nekoliko demonstracijskih primjera algoritma. (Zahtjeva da korisnik ima i UMN i UCSD bazu!)

## 2. Korištenje algoritma nad proizvoljnim datotekama

Korisnik može koristiti algoritam na dva načina. Prvi je da jednostavno pokrene **anomalydetection.py** iz komandne linije. Program će korisnika provesti kroz postavljanje parametara i postavljanje puta do datoteka. (Napomena: instrukcije u programu su na engleskom.) Alternativno, kornisnik može u svom vlastitom kodu *import*-ati metodu **anomalyDetect** iz module **anomalydetection** i direktno postaviti njene parametre.

Pri odabiru datoteka, korisnik može:
* Koristiti ili videozapis ili skup slika koje predstavljaju frame-ove videozapisa
* Pri korištenju videozapisa, korisnik opisuje put do datoteke (pri čemu uključuje i ekstenziju)
* Pri korištenju skupu slika, korisnik upisuje putanju do mape koja sadrži slike
    * Pri ovome je važno naglasiti kako slike MORAJU biti pravilno numerirane, i to u formatu gdje je uvijek predstavljena s N znamenki. Primjerice, ako je N = 3, nazivi datoteka moraju biti u obliku '001', '025', '101', itd.
    * Putanja do mape mora sadržavati *backslash* ('/') na kraju. (Primjerice: 'UCSDped1/Test/Test1/')
    * Dopušteno je da datoteke slika imaju i neka slova prije znamenaka u svojem imenu. Primjerice: 'frame_001'. Ta slova moraju, doduše, biti ista na svim slikama. Isto tako, korisnik taj prefiks navodi u putu do datoteke, direktno iza zadnjeg *backslash*-a. U ovom slučaju, to bi bilo: 'UCSDped1/Test/Test1/frame_'
   
Nakon pokretanja algoritma, korisniku će se u komandnoj liniji ispisitavti kad god se detektira anomalija, na kojem *frame*-u se detektirala anomalija, te iznos sume interakcijskih sila *outlier*-a.

Sama metoda **anomalyDetect** vraća polje, pri kojem svaki element predstavlja jedan *frame*. Ako je element 0, na tom elementu nije postojala anomalija. Ako je 1, zabilježeno je kako je postojala.

### 2. 1. Parametri algoritma

Metoda **anomalyDetect** ima sljedeći prototip:

> def anomalyDetect(foldPath, extension='', L=10, useExistingRef=False, refF=None, refScale=1.1, frameDigits=3, vidFile=False, tau=0.5)

Pri čemu su:

* *foldPath* || Putanja do datoteke videozapisa ili mape koja sadrži skup *frame*-ova. Ona mora slijediti prijašnje opisana pravila.
* *extension* || Koristi se samo ako korisnik koristi mapu slika *frame*-ova. Odnosi se na ekstenziju istih slika.
* *L* || Broj predhodnih *frame*-ova koje algoritam koristi pri računanju prosječnog optičkog toka.
* *useExistingRef* || Zastavica koja provjerava zadaje li korisnik svoj vlastiti *treshold* pri određivanju postoji li anomalija na nekom *frame*-u ili ne. Ako je postavljena na *False*, algoritam će pretpostaviti da se u prvih *L* *frame*-ova ne postoji anomalija, te koristiti sumu *outlier*-a interakcijskih sila jedne od njih kao treshold. (NE PREPORTUČA SE!)
* *refF* || Iznos *treshold*-a. Ignorira se ako je *useExistingRef=False*.
* *refScale* || Parametar koji opisuje koliko puta suma *outlier*-a interakcijskih sila nekog *frame*-a mora biti veća od *treshold*-a kako bi se smatralo da postoji anomalija.
* *frameDigits* || Koristi se samo ako korisnik koristi mapu slika *frame*-ova. Broj koji opisuje koliko znamenaka N se koristi za redni broj *frame*-a u imenu datoteke slike.
* *vidFile* || Zastavica koja provjerava koristi li korisnik videozapis ili mapu slika.
* *tau* || Iznos relaksacijskog faktora. Koristi se pri računanju interakcijskih sila. Idealno treba biti u intervalu [0.5, 0.85].

### 2. 2. Pomoćna metoda za određivanje *treshold*-a

Ako korisnik ne koristi gotovu metodu za postavljanje parametara i želi koristiti algoritam u svom kodu, od pomoći može biti metoda **getAvgInteractionForceSum**, koja se također nalazi u **anomalydetection.py**. Njen prototip glasi:

> def getAvgInteractionForceSum(foldPath, extension='', L=10, frameDigits=3, vidFile=False, tau=0.5)

Pri čemu svi parametri imaju isto značenje kao u prethodnoj metodi. 

Korisnik ovoj metodi predaje videozapis ili skup slika koje predstavljaju obično ponašanje ljudi (bez anomalija), na temelju kojeg metoda računa prosječnu sumu *outlier*-a svih *frame*-ova. Ta vrijednost se može koristiti kao *treshold*. 

## 3. Testiranje algoritma nad postojećim bazama.

Korisnik može koristiti prijašnje opisane, ugrađene evaluacije napravljene za određene baze.

Baze se mogu naći:

* UCSD Anomaly Detection Dataset: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
* UMN baza: https://drive.google.com/file/d/1Bl8CwlRqogIjct-CE7sgIKFPeS5vyIPL/view?usp=sharing

Napomena: UMN baza je na privatnom link-u jer nismo našli originalno mjesto gdje se nalazi.