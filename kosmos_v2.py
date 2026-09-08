"""
LOTO 7/39 — PATTERN RECOGNITION ZA ISTRAŽIVANJE SVEMIRA

Jedan zajednički CSV i jedna NEXT predikcija.

Model kombinuje:

1. wavelet osobine;
2. matched-filter osobine;
3. change-point detection;
4. vremenski linearni autoenkoder;
5. Hidden Markov režime;
6. grafovski model;
7. Bajesovu procenu verovatnoće i neizvesnosti;
8. hronološku walk-forward validaciju;
9. potpuno odvojen zamrznuti holdout;
10. blok-bootstrap i Monte Karlo nultu hipotezu.

Prvi red CSV fajla smatra se najstarijim.
Poslednji red CSV fajla smatra se najnovijim.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta as beta_raspodela

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as greska:
    raise SystemExit(
        "\nNedostaje paket hmmlearn.\n"
        "Instalacija:\n\n"
        "pip install hmmlearn\n"
    ) from greska


# =============================================================================
# PODEŠAVANJA
# =============================================================================

SEED = 39

BROJ_KUGLICA = 39
BROJEVA_U_KOMBINACIJI = 7

OSNOVNA_STOPA = BROJEVA_U_KOMBINACIJI / BROJ_KUGLICA
SLUCAJNO_OCEKIVANJE = BROJEVA_U_KOMBINACIJI**2 / BROJ_KUGLICA

CSV_PUTANJA = Path(
    "/Users/4c/Desktop/GHQ/data/loto7_4680_k71.csv"
)

MINIMUM_ISTORIJE = 300

BROJ_VALIDACIONIH_KORAKA = 120
BROJ_HOLDOUT_KORAKA = 160

BROJ_HMM_REZIMA = 3
HMM_ITERACIJE = 80

AUTOENKODER_PROZOR = 20
AUTOENKODER_LATENTNO = 8
AUTOENKODER_ISTORIJA = 600

GRAF_PROZOR = 500
GRAF_ITERACIJE = 40
GRAF_PRIGUSENJE = 0.85

BROJ_MONTE_KARLO_SIMULACIJA = 50_000
BROJ_BOOTSTRAP_SIMULACIJA = 20_000
BOOTSTRAP_BLOK = 10

EPS = 1e-9

NAZIVI_MODELA = (
    "Wavelet i matched filter",
    "Change-point detection",
    "Vremenski autoenkoder",
    "Hidden Markov režimi",
    "Grafovski model",
    "Bajesov model",
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# OPŠTE FUNKCIJE
# =============================================================================

def naslov(tekst: str, znak: str = "=") -> None:
    print()
    print(znak * 78)
    print(tekst)
    print(znak * 78)


def formatiraj_kombinaciju(brojevi) -> str:
    return ", ".join(f"{int(broj):02d}" for broj in sorted(brojevi))


def standardizuj(vrednosti: np.ndarray) -> np.ndarray:
    vrednosti = np.asarray(vrednosti, dtype=float)

    sredina = float(np.mean(vrednosti))
    odstupanje = float(np.std(vrednosti))

    if not np.isfinite(odstupanje) or odstupanje < EPS:
        return np.zeros_like(vrednosti)

    rezultat = (vrednosti - sredina) / odstupanje

    return np.nan_to_num(
        rezultat,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def skor_u_verovatnoce(skor: np.ndarray) -> np.ndarray:
    """
    Pretvara kontinuirani skor u 39 marginalnih verovatnoća čiji je
    zbir tačno sedam.
    """

    z = standardizuj(skor)

    donja = -30.0
    gornja = 30.0
    nagib = 0.75

    for _ in range(80):
        sredina = (donja + gornja) / 2.0
        zbir = np.sum(expit(nagib * z + sredina))

        if zbir > BROJEVA_U_KOMBINACIJI:
            gornja = sredina
        else:
            donja = sredina

    pomeraj = (donja + gornja) / 2.0
    verovatnoce = expit(nagib * z + pomeraj)

    return np.clip(verovatnoce, EPS, 1.0 - EPS)


def ucitaj_csv(putanja: Path) -> tuple[np.ndarray, np.ndarray]:
    if not putanja.exists():
        raise FileNotFoundError(f"CSV fajl ne postoji: {putanja}")

    okvir = pd.read_csv(putanja, header=None)
    okvir = okvir.dropna(axis=1, how="all")

    if okvir.shape[1] != BROJEVA_U_KOMBINACIJI:
        raise ValueError(
            f"Očekivano je tačno 7 kolona, "
            f"a pronađeno je {okvir.shape[1]}."
        )

    okvir = okvir.apply(pd.to_numeric, errors="coerce")

    if okvir.isna().any().any():
        neispravni = np.where(okvir.isna().any(axis=1))[0] + 1
        raise ValueError(
            "CSV sadrži neispravne vrednosti u redovima: "
            f"{neispravni[:20].tolist()}"
        )

    izvlacenja = okvir.to_numpy(dtype=int)

    for indeks, red in enumerate(izvlacenja, start=1):
        if np.any(red < 1) or np.any(red > BROJ_KUGLICA):
            raise ValueError(
                f"Red {indeks}: brojevi moraju biti između 1 i 39."
            )

        if len(np.unique(red)) != BROJEVA_U_KOMBINACIJI:
            raise ValueError(
                f"Red {indeks}: svih sedam brojeva mora biti različito."
            )

    matrica = np.zeros(
        (len(izvlacenja), BROJ_KUGLICA),
        dtype=np.float64,
    )

    for i, red in enumerate(izvlacenja):
        matrica[i, red - 1] = 1.0

    return izvlacenja, matrica


# =============================================================================
# 1. WAVELET I MATCHED-FILTER MODEL
# =============================================================================

def haar_poslednja_razlika(
    signal: np.ndarray,
    velicina_bloka: int,
) -> float:
    """
    Poslednji Haar detalj: razlika proseka dve susedne polovine
    poslednjeg vremenskog bloka.
    """

    potrebna_duzina = 2 * velicina_bloka

    if len(signal) < potrebna_duzina:
        return 0.0

    deo = signal[-potrebna_duzina:]

    leva = np.mean(deo[:velicina_bloka])
    desna = np.mean(deo[velicina_bloka:])

    return float(desna - leva)


def normalizovan_matched_filter(
    signal: np.ndarray,
    sablon: np.ndarray,
) -> float:
    duzina = len(sablon)

    if len(signal) < duzina:
        return 0.0

    x = signal[-duzina:].astype(float)
    x -= np.mean(x)

    t = sablon.astype(float)
    t -= np.mean(t)

    imenilac = np.linalg.norm(x) * np.linalg.norm(t)

    if imenilac < EPS:
        return 0.0

    return float(np.dot(x, t) / imenilac)


def wavelet_matched_model(matrica: np.ndarray) -> np.ndarray:
    istorija = matrica[-min(len(matrica), 512):]

    sabloni = (
        np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float),
        np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=float),
        np.array([0, 0, 0, 1, 0, 0, 0, -1], dtype=float),
        np.array([-1, 0, 0, 0, 0, 0, 0, 1], dtype=float),
    )

    skorovi = np.zeros(BROJ_KUGLICA, dtype=float)

    for broj in range(BROJ_KUGLICA):
        signal = istorija[:, broj]

        haar = (
            0.40 * haar_poslednja_razlika(signal, 2)
            + 0.30 * haar_poslednja_razlika(signal, 4)
            + 0.20 * haar_poslednja_razlika(signal, 8)
            + 0.10 * haar_poslednja_razlika(signal, 16)
        )

        matched = np.mean(
            [
                normalizovan_matched_filter(signal, sablon)
                for sablon in sabloni
            ]
        )

        skorovi[broj] = 0.60 * haar + 0.40 * matched

    return standardizuj(skorovi)


# =============================================================================
# 2. CHANGE-POINT DETECTION
# =============================================================================

def change_point_jednog_signala(
    signal: np.ndarray,
) -> tuple[float, float]:
    """
    Traži poslednju statistički najizraženiju promenu sredine.

    Vraća:
    - smer promene;
    - jačinu promene.
    """

    n = len(signal)

    if n < 60:
        return 0.0, 0.0

    minimalni_segment = 15
    najbolja_jacina = 0.0
    najbolji_smer = 0.0

    ukupna_std = float(np.std(signal))

    if ukupna_std < EPS:
        return 0.0, 0.0

    najraniji = max(minimalni_segment, n - 200)
    najkasniji = n - minimalni_segment

    for presek in range(najraniji, najkasniji + 1):
        pre = signal[max(0, presek - 80):presek]
        posle = signal[presek:]

        if len(pre) < minimalni_segment or len(posle) < minimalni_segment:
            continue

        razlika = float(np.mean(posle) - np.mean(pre))

        standardna_greska = ukupna_std * math.sqrt(
            1.0 / len(pre) + 1.0 / len(posle)
        )

        jacina = abs(razlika) / max(standardna_greska, EPS)

        # Novije promene su relevantnije od veoma starih.
        starost = n - presek
        vremenska_tezina = math.exp(-starost / 100.0)

        ponderisana_jacina = jacina * vremenska_tezina

        if ponderisana_jacina > najbolja_jacina:
            najbolja_jacina = ponderisana_jacina
            najbolji_smer = math.copysign(1.0, razlika)

    return najbolji_smer, najbolja_jacina


def change_point_model(matrica: np.ndarray) -> np.ndarray:
    istorija = matrica[-min(len(matrica), 500):]
    skorovi = np.zeros(BROJ_KUGLICA, dtype=float)

    for broj in range(BROJ_KUGLICA):
        smer, jacina = change_point_jednog_signala(
            istorija[:, broj]
        )

        # Ograničenje sprečava da jedan slučajni presek potpuno
        # preuzme završnu predikciju.
        skorovi[broj] = smer * min(jacina, 4.0)

    return standardizuj(skorovi)


# =============================================================================
# 3. VREMENSKI AUTOENKODER
# =============================================================================

def napravi_vremenske_prozore(
    matrica: np.ndarray,
    duzina: int,
) -> np.ndarray:
    if len(matrica) < duzina:
        return np.empty((0, duzina * BROJ_KUGLICA))

    prozori = []

    for kraj in range(duzina, len(matrica) + 1):
        prozor = matrica[kraj - duzina:kraj]
        prozori.append(prozor.reshape(-1))

    return np.asarray(prozori, dtype=float)


def vremenski_autoenkoder(matrica: np.ndarray) -> np.ndarray:
    """
    Stabilan linearni autoenkoder.

    Linearni autoenkoder sa kvadratnim gubitkom matematički odgovara
    PCA/SVD latentnom prostoru. Za mali Loto skup stabilniji je od
    duboke neuronske mreže sa velikim brojem parametara.
    """

    istorija = matrica[-min(len(matrica), AUTOENKODER_ISTORIJA):]

    prozori = napravi_vremenske_prozore(
        istorija,
        AUTOENKODER_PROZOR,
    )

    if len(prozori) < 30:
        return np.zeros(BROJ_KUGLICA)

    sredina = np.mean(prozori, axis=0)
    centrirano = prozori - sredina

    _, _, vt = np.linalg.svd(
        centrirano,
        full_matrices=False,
    )

    rang = min(
        AUTOENKODER_LATENTNO,
        len(vt),
        len(prozori) - 1,
    )

    komponente = vt[:rang]

    poslednji = centrirano[-1]
    latentno = poslednji @ komponente.T
    rekonstrukcija = latentno @ komponente + sredina

    original = prozori[-1].reshape(
        AUTOENKODER_PROZOR,
        BROJ_KUGLICA,
    )

    rekonstruisano = rekonstrukcija.reshape(
        AUTOENKODER_PROZOR,
        BROJ_KUGLICA,
    )

    greska = original - rekonstruisano

    vremenske_tezine = np.linspace(
        0.25,
        1.0,
        AUTOENKODER_PROZOR,
    )
    vremenske_tezine /= vremenske_tezine.sum()

    # Potpisana anomalija pokazuje smer odstupanja, a apsolutna
    # vrednost njegov intenzitet.
    potpisana = vremenske_tezine @ greska
    intenzitet = vremenske_tezine @ np.abs(greska)

    skor = potpisana + 0.20 * intenzitet

    return standardizuj(skor)


# =============================================================================
# 4. HIDDEN MARKOV REŽIMI
# =============================================================================

def osobine_izvlacenja(matrica: np.ndarray) -> np.ndarray:
    brojevi = np.arange(1, BROJ_KUGLICA + 1, dtype=float)

    zbir = matrica @ brojevi
    neparni = matrica @ (brojevi % 2)
    niski = matrica[:, :19].sum(axis=1)

    sredina = zbir / BROJEVA_U_KOMBINACIJI

    centrirani = (
        brojevi[None, :] - sredina[:, None]
    ) * matrica

    rasipanje = np.sqrt(
        np.sum(centrirani**2, axis=1)
        / BROJEVA_U_KOMBINACIJI
    )

    osobine = np.column_stack(
        [
            zbir,
            neparni,
            niski,
            rasipanje,
        ]
    )

    prosek = osobine.mean(axis=0)
    std = osobine.std(axis=0)
    std[std < EPS] = 1.0

    return (osobine - prosek) / std


def hmm_model(matrica: np.ndarray) -> np.ndarray:
    if len(matrica) < 150:
        return np.zeros(BROJ_KUGLICA)

    istorija = matrica[-min(len(matrica), 800):]
    osobine = osobine_izvlacenja(istorija)

    model = GaussianHMM(
        n_components=BROJ_HMM_REZIMA,
        covariance_type="diag",
        n_iter=HMM_ITERACIJE,
        tol=1e-3,
        random_state=SEED,
        min_covar=1e-4,
    )

    try:
        model.fit(osobine)
        stanja = model.predict(osobine)
    except Exception:
        return np.zeros(BROJ_KUGLICA)

    poslednje_stanje = int(stanja[-1])
    sledece_tezine = model.transmat_[poslednje_stanje]

    rezultat = np.zeros(BROJ_KUGLICA, dtype=float)

    for stanje in range(BROJ_HMM_REZIMA):
        maska = stanja == stanje
        broj_redova = int(np.sum(maska))

        if broj_redova == 0:
            stopa = np.full(
                BROJ_KUGLICA,
                OSNOVNA_STOPA,
            )
        else:
            brojanja = istorija[maska].sum(axis=0)

            prior = max(20.0, broj_redova * 0.20)

            stopa = (
                brojanja + prior * OSNOVNA_STOPA
            ) / (
                broj_redova + prior
            )

        rezultat += sledece_tezine[stanje] * stopa

    return standardizuj(
        rezultat - OSNOVNA_STOPA
    )


# =============================================================================
# 5. GRAFOVSKI MODEL
# =============================================================================

def grafovski_model(matrica: np.ndarray) -> np.ndarray:
    istorija = matrica[-min(len(matrica), GRAF_PROZOR):]
    n = len(istorija)

    starost = np.arange(n - 1, -1, -1, dtype=float)
    vremenske_tezine = np.exp(
        -math.log(2.0) * starost / max(50.0, n / 3.0)
    )

    ponderisana = istorija * np.sqrt(vremenske_tezine[:, None])
    susedstvo = ponderisana.T @ ponderisana

    np.fill_diagonal(susedstvo, 0.0)

    # Stabilizacija grafa slabim uniformnim priorom.
    susedstvo += 0.01

    zbir_reda = susedstvo.sum(axis=1, keepdims=True)
    prelazi = susedstvo / np.clip(zbir_reda, EPS, None)

    poslednji = istorija[-1]
    personalizacija = np.full(
        BROJ_KUGLICA,
        0.30 / BROJ_KUGLICA,
    )

    personalizacija += (
        0.70
        * poslednji
        / max(float(poslednji.sum()), EPS)
    )

    rang = personalizacija.copy()

    for _ in range(GRAF_ITERACIJE):
        novi = (
            GRAF_PRIGUSENJE * prelazi.T @ rang
            + (1.0 - GRAF_PRIGUSENJE) * personalizacija
        )

        if np.max(np.abs(novi - rang)) < 1e-10:
            rang = novi
            break

        rang = novi

    return standardizuj(rang)


# =============================================================================
# 6. BAJESOV MODEL I NEIZVESNOST
# =============================================================================

def bajesov_model(
    matrica: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Beta posterior za svaki broj sa priorom centriranim na 7/39.

    Vraća:
    - završni skor;
    - posterior sredinu;
    - širinu intervala pouzdanosti.
    """

    istorija = matrica[-min(len(matrica), 500):]
    n = len(istorija)

    starost = np.arange(n - 1, -1, -1, dtype=float)
    vremenske_tezine = np.exp(
        -math.log(2.0) * starost / max(50.0, n / 3.0)
    )

    uspesi = vremenske_tezine @ istorija
    efektivni_n = float(vremenske_tezine.sum())

    jacina_priora = efektivni_n

    alfa_prior = jacina_priora * OSNOVNA_STOPA
    beta_prior = jacina_priora * (1.0 - OSNOVNA_STOPA)

    alfa = alfa_prior + uspesi
    beta_parametar = (
        beta_prior + efektivni_n - uspesi
    )

    sredina = alfa / (alfa + beta_parametar)

    donja = beta_raspodela.ppf(
        0.025,
        alfa,
        beta_parametar,
    )

    gornja = beta_raspodela.ppf(
        0.975,
        alfa,
        beta_parametar,
    )

    sirina = gornja - donja

    # Broj dobija veći skor samo ako je posterior povoljan i dovoljno
    # siguran. Neizvesnost se kažnjava.
    skor = (
        sredina
        - OSNOVNA_STOPA
        - 0.20 * sirina
    )

    return standardizuj(skor), sredina, sirina


# =============================================================================
# SVI MODELI
# =============================================================================

def izracunaj_modele(
    matrica: np.ndarray,
) -> tuple[np.ndarray, dict]:
    bajes_skor, bajes_sredina, bajes_sirina = bajesov_model(
        matrica
    )

    skorovi = np.vstack(
        [
            wavelet_matched_model(matrica),
            change_point_model(matrica),
            vremenski_autoenkoder(matrica),
            hmm_model(matrica),
            grafovski_model(matrica),
            bajes_skor,
        ]
    )

    verovatnoce = np.vstack(
        [
            skor_u_verovatnoce(skor)
            for skor in skorovi
        ]
    )

    dodatno = {
        "bajes_sredina": bajes_sredina,
        "bajes_sirina": bajes_sirina,
    }

    return verovatnoce, dodatno


def kombinuj_modele(
    modeli: np.ndarray,
    tezine: np.ndarray,
) -> np.ndarray:
    rezultat = np.average(
        modeli,
        axis=0,
        weights=tezine,
    )

    rezultat *= (
        BROJEVA_U_KOMBINACIJI
        / max(float(rezultat.sum()), EPS)
    )

    return np.clip(rezultat, EPS, 1.0 - EPS)


def izaberi_sedam(verovatnoce: np.ndarray) -> np.ndarray:
    indeksi = np.argsort(
        verovatnoce,
        kind="stable",
    )[-BROJEVA_U_KOMBINACIJI:]

    return np.sort(indeksi + 1)


# =============================================================================
# WALK-FORWARD VALIDACIJA
# =============================================================================

def napravi_walk_forward(
    matrica: np.ndarray,
    pocetak: int,
    kraj: int,
    naziv: str,
) -> tuple[np.ndarray, np.ndarray]:
    predikcije = []
    ishodi = []

    ukupno = kraj - pocetak

    for redni, granica in enumerate(
        range(pocetak, kraj),
        start=1,
    ):
        modeli, _ = izracunaj_modele(
            matrica[:granica]
        )

        predikcije.append(modeli)
        ishodi.append(matrica[granica])

        if (
            redni == 1
            or redni == ukupno
            or redni % 10 == 0
        ):
            print(f"  {naziv}: {redni}/{ukupno}")

    return (
        np.asarray(predikcije, dtype=float),
        np.asarray(ishodi, dtype=float),
    )


def funkcija_gubitka(
    tezine: np.ndarray,
    predikcije: np.ndarray,
    ishodi: np.ndarray,
) -> float:
    kombinovano = np.einsum(
        "tmn,m->tn",
        predikcije,
        tezine,
    )

    brier = float(
        np.mean((kombinovano - ishodi) ** 2)
    )

    # Regularizacija sprečava da slučajan rezultat jednog modela
    # dobije svu težinu.
    ravnomerno = np.full(
        len(tezine),
        1.0 / len(tezine),
    )

    kazna = 0.002 * float(
        np.sum((tezine - ravnomerno) ** 2)
    )

    return brier + kazna


def odredi_tezine(
    predikcije: np.ndarray,
    ishodi: np.ndarray,
) -> np.ndarray:
    broj_modela = predikcije.shape[1]

    pocetne = np.full(
        broj_modela,
        1.0 / broj_modela,
    )

    rezultat = minimize(
        funkcija_gubitka,
        pocetne,
        args=(predikcije, ishodi),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * broj_modela,
        constraints={
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
        },
        options={
            "maxiter": 500,
            "ftol": 1e-12,
        },
    )

    if not rezultat.success:
        return pocetne

    tezine = np.clip(
        rezultat.x,
        0.0,
        1.0,
    )

    if tezine.sum() < EPS:
        return pocetne

    return tezine / tezine.sum()


# =============================================================================
# HOLDOUT OCENA
# =============================================================================

def oceni_holdout(
    predikcije: np.ndarray,
    ishodi: np.ndarray,
    tezine: np.ndarray,
) -> dict:
    kombinovano = np.einsum(
        "tmn,m->tn",
        predikcije,
        tezine,
    )

    pogodci = []
    brier = []

    for verovatnoce, stvarno in zip(
        kombinovano,
        ishodi,
    ):
        izbor = np.argsort(
            verovatnoce,
            kind="stable",
        )[-BROJEVA_U_KOMBINACIJI:]

        pogodci.append(
            int(stvarno[izbor].sum())
        )

        brier.append(
            float(
                np.mean(
                    (verovatnoce - stvarno) ** 2
                )
            )
        )

    pogodci = np.asarray(pogodci, dtype=int)

    return {
        "pogodci": pogodci,
        "prosek": float(np.mean(pogodci)),
        "brier": float(np.mean(brier)),
        "najmanje_3": float(np.mean(pogodci >= 3)),
        "najmanje_4": float(np.mean(pogodci >= 4)),
        "maksimum": int(np.max(pogodci)),
    }


# =============================================================================
# BLOK-BOOTSTRAP
# =============================================================================

def blok_bootstrap(
    pogodci: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    pogodci = np.asarray(pogodci, dtype=float)
    n = len(pogodci)

    broj_blokova = math.ceil(
        n / BOOTSTRAP_BLOK
    )

    maksimalni_pocetak = max(
        1,
        n - BOOTSTRAP_BLOK + 1,
    )

    proseci = np.empty(
        BROJ_BOOTSTRAP_SIMULACIJA,
        dtype=float,
    )

    for simulacija in range(
        BROJ_BOOTSTRAP_SIMULACIJA
    ):
        delovi = []

        for _ in range(broj_blokova):
            pocetak = int(
                rng.integers(
                    0,
                    maksimalni_pocetak,
                )
            )

            delovi.append(
                pogodci[
                    pocetak:
                    pocetak + BOOTSTRAP_BLOK
                ]
            )

        uzorak = np.concatenate(delovi)[:n]
        proseci[simulacija] = np.mean(uzorak)

    donja, gornja = np.quantile(
        proseci,
        [0.025, 0.975],
    )

    return float(donja), float(gornja)


# =============================================================================
# MONTE KARLO NULTA HIPOTEZA
# =============================================================================

def monte_karlo_test(
    broj_koraka: int,
    stvarni_prosek: float,
    rng: np.random.Generator,
) -> dict:
    simulirani_pogodci = rng.hypergeometric(
        ngood=BROJEVA_U_KOMBINACIJI,
        nbad=BROJ_KUGLICA - BROJEVA_U_KOMBINACIJI,
        nsample=BROJEVA_U_KOMBINACIJI,
        size=(
            BROJ_MONTE_KARLO_SIMULACIJA,
            broj_koraka,
        ),
    )

    simulirani_proseci = np.mean(
        simulirani_pogodci,
        axis=1,
    )

    p_vrednost = (
        np.sum(
            simulirani_proseci >= stvarni_prosek
        ) + 1
    ) / (
        BROJ_MONTE_KARLO_SIMULACIJA + 1
    )

    donja, gornja = np.quantile(
        simulirani_proseci,
        [0.025, 0.975],
    )

    return {
        "p_vrednost": float(p_vrednost),
        "sredina": float(
            np.mean(simulirani_proseci)
        ),
        "donja": float(donja),
        "gornja": float(gornja),
    }


# =============================================================================
# GLAVNA OBRADA
# =============================================================================

def obradi() -> dict:
    izvlacenja, matrica = ucitaj_csv(
        CSV_PUTANJA
    )

    n = len(matrica)

    naslov("OBRADA ZAJEDNIČKOG CSV FAJLA")

    print(f"CSV: {CSV_PUTANJA}")
    print(f"Broj redova: {n}")
    print("Prvi red se tretira kao najstariji.")
    print("Poslednji red se tretira kao najnoviji.")

    potrebno = (
        MINIMUM_ISTORIJE
        + BROJ_VALIDACIONIH_KORAKA
        + BROJ_HOLDOUT_KORAKA
    )

    if n < potrebno:
        raise RuntimeError(
            f"Potrebno je najmanje {potrebno} redova, "
            f"a pronađeno je {n}."
        )

    holdout_pocetak = (
        n - BROJ_HOLDOUT_KORAKA
    )

    validacija_pocetak = (
        holdout_pocetak
        - BROJ_VALIDACIONIH_KORAKA
    )

    print()
    print("Razvojna hronološka validacija...")

    razvoj_predikcije, razvoj_ishodi = napravi_walk_forward(
        matrica=matrica,
        pocetak=validacija_pocetak,
        kraj=holdout_pocetak,
        naziv="Razvoj",
    )

    tezine = odredi_tezine(
        razvoj_predikcije,
        razvoj_ishodi,
    )

    print()
    print("Zamrznuta holdout provera...")

    holdout_predikcije, holdout_ishodi = napravi_walk_forward(
        matrica=matrica,
        pocetak=holdout_pocetak,
        kraj=n,
        naziv="Holdout",
    )

    holdout = oceni_holdout(
        holdout_predikcije,
        holdout_ishodi,
        tezine,
    )

    rng = np.random.default_rng(SEED)

    bootstrap_donja, bootstrap_gornja = blok_bootstrap(
        holdout["pogodci"],
        rng,
    )

    monte_karlo = monte_karlo_test(
        broj_koraka=len(holdout["pogodci"]),
        stvarni_prosek=holdout["prosek"],
        rng=rng,
    )

    zavrsni_modeli, dodatno = izracunaj_modele(
        matrica
    )

    zavrsne_verovatnoce = kombinuj_modele(
        zavrsni_modeli,
        tezine,
    )

    next_kombinacija = izaberi_sedam(
        zavrsne_verovatnoce
    )

    statisticki_pouzdan = (
        holdout["prosek"] > SLUCAJNO_OCEKIVANJE
        and bootstrap_donja > SLUCAJNO_OCEKIVANJE
        and monte_karlo["p_vrednost"] < 0.05
    )

    return {
        "broj_redova": n,
        "next": next_kombinacija,
        "tezine": tezine,
        "holdout": holdout,
        "bootstrap_donja": bootstrap_donja,
        "bootstrap_gornja": bootstrap_gornja,
        "monte_karlo": monte_karlo,
        "statisticki_pouzdan": statisticki_pouzdan,
        "verovatnoce": zavrsne_verovatnoce,
        "bajes_sredina": dodatno["bajes_sredina"],
        "bajes_sirina": dodatno["bajes_sirina"],
    }


# =============================================================================
# ISPIS REZULTATA
# =============================================================================

def ispisi_rezultat(rezultat: dict) -> None:
    naslov("KONAČNA NEXT PREDIKCIJA", znak="#")

    print(
        f"NEXT: {formatiraj_kombinaciju(rezultat['next'])}"
    )
    print(f"CSV redova: {rezultat['broj_redova']}")

    print()
    print("Težine modela:")

    for naziv, tezina in zip(
        NAZIVI_MODELA,
        rezultat["tezine"],
    ):
        print(f"  {naziv:<30} {tezina:>8.2%}")

    print()
    print("Sedam najbolje ocenjenih brojeva:")

    poredak = np.argsort(
        rezultat["verovatnoce"],
        kind="stable",
    )[::-1][:BROJEVA_U_KOMBINACIJI]

    for mesto, indeks in enumerate(
        poredak,
        start=1,
    ):
        print(
            f"  {mesto}. broj {indeks + 1:02d}"
            f" — ensemble {rezultat['verovatnoce'][indeks]:.6f}"
            f" — Bajes {rezultat['bajes_sredina'][indeks]:.6f}"
            f" — širina 95% intervala "
            f"{rezultat['bajes_sirina'][indeks]:.6f}"
        )

    holdout = rezultat["holdout"]
    monte_karlo = rezultat["monte_karlo"]

    print()
    print("Zamrznuta holdout provera:")
    print(
        f"  Broj koraka:                    "
        f"{len(holdout['pogodci'])}"
    )
    print(
        f"  Prosečan broj pogodaka:         "
        f"{holdout['prosek']:.6f}"
    )
    print(
        f"  Slučajno očekivanje:            "
        f"{SLUCAJNO_OCEKIVANJE:.6f}"
    )
    print(
        f"  Brierov skor:                   "
        f"{holdout['brier']:.6f}"
    )
    print(
        f"  Najmanje tri pogotka:           "
        f"{holdout['najmanje_3']:.2%}"
    )
    print(
        f"  Najmanje četiri pogotka:        "
        f"{holdout['najmanje_4']:.2%}"
    )
    print(
        f"  Najveći broj pogodaka:          "
        f"{holdout['maksimum']}"
    )
    print(
        f"  Blok-bootstrap 95% interval:    "
        f"[{rezultat['bootstrap_donja']:.6f}, "
        f"{rezultat['bootstrap_gornja']:.6f}]"
    )

    print()
    print("Monte Karlo nulta hipoteza:")
    print(
        f"  Broj simulacija:                "
        f"{BROJ_MONTE_KARLO_SIMULACIJA:,}"
    )
    print(
        f"  Prosek poštenog modela:         "
        f"{monte_karlo['sredina']:.6f}"
    )
    print(
        f"  95% opseg poštenog modela:      "
        f"[{monte_karlo['donja']:.6f}, "
        f"{monte_karlo['gornja']:.6f}]"
    )
    print(
        f"  Jednostrana p-vrednost:         "
        f"{monte_karlo['p_vrednost']:.6f}"
    )

    print()
    print("GLAVNI ODGOVOR:")

    if rezultat["statisticki_pouzdan"]:
        print(
            "  DA — pronađena je statistički pouzdana prediktivna "
            "informacija na zamrznutom holdoutu."
        )
        print(
            "  Donja granica 95% intervala prelazi slučajno "
            f"očekivanje od {SLUCAJNO_OCEKIVANJE:.6f},"
        )
        print(
            "  a Monte Karlo jednostrana p-vrednost manja je od 0.05."
        )
    else:
        print(
            "  NE — istorijska izvlačenja nisu pokazala statistički "
            "pouzdanu prediktivnu prednost"
        )
        print(
            "  iznad slučajnog očekivanja od "
            f"{SLUCAJNO_OCEKIVANJE:.6f} pogodaka."
        )

    print()
    print(
        "NEXT kombinacija je rezultat šest modela za prepoznavanje "
        "obrazaca, ali nije dokaz"
    )
    print(
        "da se buduće pošteno izvlačenje može pouzdano predvideti."
    )


# =============================================================================
# POKRETANJE
# =============================================================================

def main() -> None:
    np.random.seed(SEED)

    naslov(
        "LOTO 7/39 — SVEMIRSKI PATTERN RECOGNITION SISTEM"
    )

    print(f"Seed: {SEED}")
    print(f"Teorijska stopa broja: {OSNOVNA_STOPA:.9f}")
    print(
        f"Teorijsko očekivanje pogodaka: "
        f"{SLUCAJNO_OCEKIVANJE:.9f}"
    )
    print(
        f"Ukupno mogućih kombinacija: "
        f"{math.comb(BROJ_KUGLICA, BROJEVA_U_KOMBINACIJI):,}"
    )

    rezultat = obradi()
    ispisi_rezultat(rezultat)


if __name__ == "__main__":
    main()



"""
==============================================================================
LOTO 7/39 — SVEMIRSKI PATTERN RECOGNITION SISTEM
==============================================================================
Seed: 39
Teorijska stopa broja: 0.179487179
Teorijsko očekivanje pogodaka: 1.256410256
Ukupno mogućih kombinacija: 15,380,937

==============================================================================
OBRADA ZAJEDNIČKOG CSV FAJLA
==============================================================================
CSV: /Users/4c/Desktop/GHQ/data/loto7_4680_k71.csv
Broj redova: 4680
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.

Razvojna hronološka validacija...
  Razvoj: 1/120
  Razvoj: 10/120
  Razvoj: 20/120
  Razvoj: 30/120
  Razvoj: 40/120
  Razvoj: 50/120
  Razvoj: 60/120
  Razvoj: 70/120
  Razvoj: 80/120
  Razvoj: 90/120
  Razvoj: 100/120
  Razvoj: 110/120
  Razvoj: 120/120

Zamrznuta holdout provera...
  Holdout: 1/160
  Holdout: 10/160
  Holdout: 20/160
  Holdout: 30/160
  Holdout: 40/160
  Holdout: 50/160
  Holdout: 60/160
  Holdout: 70/160
  Holdout: 80/160
  Holdout: 90/160
  Holdout: 100/160
  Holdout: 110/160
  Holdout: 120/160
  Holdout: 130/160
  Holdout: 140/160
  Holdout: 150/160
  Holdout: 160/160

##############################################################################
KONAČNA NEXT PREDIKCIJA
##############################################################################
NEXT: 08, 10, 21, 22, 29, 31, 36
CSV redova: 4680

Težine modela:
  Wavelet i matched filter         28.69%
  Change-point detection           35.45%
  Vremenski autoenkoder             2.20%
  Hidden Markov režimi             23.73%
  Grafovski model                   0.00%
  Bajesov model                     9.93%

Sedam najbolje ocenjenih brojeva:
  1. broj 08 — ensemble 0.337364 — Bajes 0.193629 — širina 95% intervala 0.075266
  2. broj 36 — ensemble 0.276748 — Bajes 0.173221 — širina 95% intervala 0.072076
  3. broj 22 — ensemble 0.274287 — Bajes 0.182792 — širina 95% intervala 0.073615
  4. broj 31 — ensemble 0.268903 — Bajes 0.189603 — širina 95% intervala 0.074664
  5. broj 10 — ensemble 0.264368 — Bajes 0.189051 — širina 95% intervala 0.074580
  6. broj 21 — ensemble 0.262749 — Bajes 0.173261 — širina 95% intervala 0.072083
  7. broj 29 — ensemble 0.255784 — Bajes 0.185305 — širina 95% intervala 0.074006

Zamrznuta holdout provera:
  Broj koraka:                    160
  Prosečan broj pogodaka:         1.143750
  Slučajno očekivanje:            1.256410
  Brierov skor:                   0.151599
  Najmanje tri pogotka:           7.50%
  Najmanje četiri pogotka:        0.00%
  Najveći broj pogodaka:          3
  Blok-bootstrap 95% interval:    [0.975000, 1.275000]

Monte Karlo nulta hipoteza:
  Broj simulacija:                50,000
  Prosek poštenog modela:         1.256391
  95% opseg poštenog modela:      [1.112500, 1.400000]
  Jednostrana p-vrednost:         0.943041

GLAVNI ODGOVOR:
  NE — istorijska izvlačenja nisu pokazala statistički pouzdanu prediktivnu prednost
  iznad slučajnog očekivanja od 1.256410 pogodaka.

NEXT kombinacija je rezultat šest modela za prepoznavanje obrazaca, ali nije dokaz
da se buduće pošteno izvlačenje može pouzdano predvideti.
"""



"""
Pattern recognition u istraživanju svemira nije jedan tajni „NASA model“, već skup statističkih i AI metoda za nalaženje pravilnosti u slikama, spektrima, signalima i telemetriji.

Najvažnije metode su:
- CNN i Vision Transformer modeli — prepoznaju kratere, stene, oblake, asteroide, galaksije i tragove objekata na snimcima.
- Autoenkoderi, Isolation Forest i Self-Organizing Maps — otkrivaju neobične događaje i kvarove koji nisu unapred označeni. NASA razvija SOM/topološke mape za grupisanje normalnog ponašanja sistema i detekciju anomalija. NASA TechPort
- LSTM, GRU i vremenski transformeri — analiziraju telemetriju, promene zračenja, Sunčev vetar i vremenski sled merenja. NASA ima javno objavljen LSTM sistem za anomalije u telemetriji letelica. NASA Software Catalog
- Matched filtering i wavelet analiza — prepoznaju veoma slabe signale poznatog oblika u velikoj količini šuma.
- Bajesovi modeli i Kalman/particle filteri — procenjuju položaj, putanju i neizvesnost na osnovu više senzora.
- Grafovske neuronske mreže — obrađuju odnose između zvezda, galaksija, senzora, satelita ili delova letelice.
- Change-point detection — pronalazi trenutak kada se ponašanje signala ili sistema promenilo.
- Klasterovanje i redukcija dimenzionalnosti — PCA, robust PCA, UMAP i slične metode izdvajaju strukturu iz ogromnih skupova merenja.
- Multimodalne neuronske mreže — zajedno koriste slike, spektar i druge fizičke osobine. Kineska akademija nauka je objavila sistem koji spaja morfologiju i spektralnu raspodelu energije radi klasifikacije zvezda, galaksija i kvazara. Chinese Academy of Sciences
- Učenje sa fizičkim ograničenjima — model ne uči samo podatke nego mora poštovati poznate fizičke zakone.
- Autonomno odlučivanje — letelica sama prepoznaje zanimljiv događaj, menja cilj posmatranja i bira koje podatke šalje na Zemlju. NASA-in Autonomous Sciencecraft Experiment koristi ugrađeno mašinsko učenje i prepoznavanje obrazaca upravo za to. NASA/JPL

NASA je, na primer, prepoznavanjem obrazaca u više od 30.000 Hubble snimaka pronašla 1.031 ranije neotkriven asteroid. NASA Science JPL razvija i autonomne sisteme koji klasifikuju teren, procenjuju opasnost i planiraju kretanje robota bez stalnog upravljanja sa Zemlje. JPL autonomous systems
Rusija i Kina koriste iste osnovne matematičke porodice metoda. Razlike su prvenstveno u instrumentima, podacima, računarima i konkretnim misijama, a ne u nekom potpuno drugačijem algoritmu. O njihovim vojnim ili poverljivim sistemima nema dovoljno proverljivih javnih podataka.

Za numeričke CSV podatke najprenosiviji deo ove „svemirske“ metodologije bio bi:
wavelet/matched-filter osobine + change-point detection + vremenski autoenkoder za anomalije + Hidden Markov režimi + grafovski model + Bajesova procena neizvesnosti.
Ključna razlika je što u astronomiji obrazac često potiče od stvarnog fizičkog izvora, dok kod pravilno izvedenog Loto izvlačenja pronađeni obrazac najčešće predstavlja statistički šum. Zato je zamrznuti hronološki holdout važniji od složenosti samog modela.
"""
