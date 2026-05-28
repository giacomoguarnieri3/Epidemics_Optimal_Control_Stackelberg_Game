"""
Modello Epidemiologico SIR — Stackelberg Game (Governo vs Cittadini)
====================================================================

Struttura gerarchica:
1. Governo (leader) sceglie c_s, spesa pubblica per controllo.
2. Cittadini (followers) osservano c_s e scelgono socialita x_t^*.
3. Dinamica SIR evolve in base a x_t^*.

Questo file estende deterministic.py aggiungendo il layer comportamentale dei cittadini.

NOTA TEORICA SUL PROCESSO DI OTTIMIZZAZIONE
-------------------------------------------

Il modello e gerarchico:
1. Il governo sceglie una spesa di controllo c_s.
2. La spesa c_s viene tradotta in una prescrizione di socialita x_bar(c_s).
3. I cittadini osservano x_bar(c_s) e il rischio percepito p_t, poi scelgono la
   propria socialita effettiva x_t^*.
4. La dinamica SIR evolve usando x_t^* nel termine di contagio.
5. Il governo valuta il costo totale previsto sulla finestra e sceglie il c_s che lo minimizza.

1) Mappa controllo -> prescrizione sociale

Nel codice si usa la forma logistica:

    x_bar(c_s) = 1 / (1 + kappa * c_s)

Perche: all'aumentare di c_s la prescrizione diventa piu restrittiva, ma resta
sempre compresa in [0, 1]. Se c_s = 0, allora x_bar = 1 e non c'e restrizione.

2) Rischio percepito dai cittadini

    p_t = clip(rho_rischio * I_t / N, 0, 1)

Perche: i cittadini non reagiscono solo alla prescrizione, ma anche allo stato
epidemico corrente. Se gli infetti aumentano, aumenta il rischio percepito.

3) Best response dei cittadini

Nel caso quadratico, la funzione di utilita e:

    U_t(x) = b0 * x - lambda * p_t * x - (q / 2) * x^2 - (eta / 2) * (x - x_bar)^2

La condizione del primo ordine produce:

    x_t^* = (b0 - lambda * p_t + eta * x_bar) / (q + eta)

Poi si applica clipping in [0, 1].

Perche:
- b0 * x rappresenta il beneficio privato della socialita.
- lambda * p_t * x rappresenta il costo sanitario percepito.
- (q / 2) * x^2 introduce rendimenti marginali decrescenti della socialita.
- (eta / 2) * (x - x_bar)^2 misura il costo di non conformarsi alla prescrizione.

Quindi c_s entra in x_t^* indirettamente tramite x_bar(c_s): se il governo aumenta
c_s, allora x_bar diminuisce e, a parita di rischio, tende a diminuire anche x_t^*.

4) Ingresso di x_t^* nelle equazioni di diffusione

La socialita effettiva non entra da sola, ma tramite un fattore di contatto:

    f(x_t^*) = (x_t^*)^potenza

Nel file la simulazione usa potenza = 1.0, quindi:

    f(x_t^*) = x_t^*

Le nuove infezioni diventano:

    new_infections_t = f(x_t^*) * beta * S_t * I_t / N

e quindi:

    S_(t+1) = S_t - new_infections_t
    I_(t+1) = I_t + new_infections_t - gamma * I_t
    R_(t+1) = R_t + gamma * I_t

Perche usare f(x) = x^potenza:
- con potenza = 1 il contatto cresce linearmente con la socialita;
- con potenza > 1 il contagio cresce in modo convesso, cioe piccole riduzioni di
  socialita possono abbattere piu che proporzionalmente i contatti, utile se si
  vuole rappresentare matching o aggregazioni sociali dense;
- con potenza < 1 si otterrebbe invece una risposta concava.

La potenza e quindi un parametro strutturale che lega comportamento individuale e
intensita dei contatti epidemiologici.

5) Come c_s entra nella funzione costo del governo

Sulla finestra di previsione, il governo valuta:

    J(c_s) = costo_epidemico_previsto(c_s) + costo_controllo(c_s)

Nel codice:

    costo_epidemico_previsto(c_s) = sum_t [ c_i * I_t + k * exp(alpha * I_t) ]
    costo_controllo(c_s) = lambda_reg_controllo * c_s^2 * (orizzonte + 1)

Nella versione attuale del file, c_s NON entra direttamente nel costo di
stato istantaneo tramite un termine proporzionale ai suscettibili. Entra invece:

- indirettamente, modificando la traiettoria prevista di I_t tramite
  c_s -> x_bar -> x_t^* -> new_infections_t;
- direttamente, tramite la penalizzazione quadratica lambda_reg_controllo * c_s^2.

6) Problema di ottimizzazione risolto dal governo

Il governo esegue una grid search su c_s in [c_min, c_max]. Per ogni candidato:

    a) calcola x_bar(c_s)
    b) simula la finestra SIR con la best response dei cittadini
    c) calcola il costo totale J(c_s)

Poi seleziona:

    c_s^* = argmin_{c_s in [c_min, c_max]} J(c_s)

Perche: il governo internalizza la risposta comportamentale dei cittadini e non
ottimizza direttamente sul solo stato epidemiologico, ma sul costo complessivo che
bilancia danno epidemico e costo della policy.

7) Stato operativo corrente del file

La versione attuale del file distingue due profili di lavoro:

- assetto prudente: simulazione base attiva, scansioni di calibrazione spente;
- assetto target-driven: attivazione delle scansioni su parametri, compresa la
    calibrazione a due stadi su (lambda_reg_controllo, c_max) per avvicinare il
    picco infetti a un benchmark prefissato.

Le funzioni di scansione restano nel file per riuso e confronto, ma sono trattate
come strumenti di analisi e non come parte della simulazione base.
"""

# =============================================================================
# 1. Import delle librerie
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

print("Librerie importate con successo.")

MAX_FLOAT = np.finfo(float).max
MAX_LOG_FLOAT = np.log(MAX_FLOAT)

# =============================================================================
# 2. Parametri del modello (stessi di deterministic.py + nuovi per Stackelberg)
# =============================================================================

# Parametri epidemiologici (BASE)
N        = 10_000     # Popolazione totale
I_t0     = 10         # Infetti iniziali
R_t0     = 0          # Guariti iniziali
S_t0     = N - I_t0 - R_t0  # Suscettibili iniziali

beta     = 0.3        # Tasso di trasmissione
gamma    = 0.05       # Tasso di guarigione

# Coefficienti del costo epidemiologico giornaliero — RICALIBRAGO PER EVITARE OVERFLOW
costo_infetto_giornaliero = 10.0  # c_i (ridotto da 1000 per evitare overflow)

# Parametri di saturazione ospedaliera — RICALIBRAGI PER EVITARE OVERFLOW
k_saturazione_ospedali = 10.0  # (ridotto da 50000 per evitare overflow)
alpha_saturazione_ospedali = 0.01  # (ridotto da 6.0 per evitare overflow)
capacita_ospedaliera_infetti = 250.0
cap_argomento_exp_saturazione_default = 50.0

# Parametri del compartimento ospedalizzati H_t
H_t0_default = 0.0
h_ospedalizzazione_default = 0.04
tau_ospedalizzazione_default = 9
degenza_media_ospedaliera_default = 9.0
gamma_ospedaliera_default = 1.0 / degenza_media_ospedaliera_default

# Parametro di efficacia della policy (vecchio modello, mantenuto per retro-compatibilita)
m_controllo = 2.0

# Peso della penalizzazione quadratica del controllo
lambda_reg_controllo = 14.0

# Parametri simulazione controllo periodico (MPC)
intervallo_controllo_default = 80
orizzonte_predizione_default = 80
c_min_default = 0.05
c_max_default = 80.0
c_iniziale_default = c_min_default
soglia_attivazione_controllo_default = 0.005
fattore_isteresi_default = 0.3
verbose_progress_default = True
mostra_grafici_default = True

# Parametri numerici
num_grid_points_default = 401
step_percent_progress_default = 5

# Flag di assetto: True = benchmark prudente corrente, False = assetto target-driven
assetto_prudente_default = True

# Parametri scansione calibrazione (alpha, lambda): utili per confronti di sensibilita
esegui_scansione_parametri_default = False
alpha_scan_default = [2.0, 3.0, 4.0, 5.0, 6.0]
lambda_scan_default = [20.0, 40.0, 80.0, 120.0]

# Parametri scansione calibrazione (soglia attivazione, isteresi): tarano l'ingresso/uscita del controllo
esegui_scansione_trigger_default = False
soglia_scan_default = [0.002, 0.0035, 0.005, 0.0075, 0.01]
isteresi_scan_default = [0.2, 0.3, 0.4, 0.5, 0.7]

# Parametri scansione calibrazione (kappa prescrizione, compliance, rischio): descrivono i follower
esegui_scansione_comportamento_default = False
kappa_scan_default = [0.04, 0.05, 0.06]
eta_scan_default = [10.0, 12.0, 14.0]
rho_scan_default = [1.0, 1.5, 2.0]

# Parametri scansione target picco (lambda, c_max): usati nel benchmark a due stadi
esegui_scansione_target_picco_default = False
target_picco_percent_default = 10.0
tolleranza_target_percent_default = 1.0
lambda_target_scan_default = [20.0, 40.0, 60.0, 80.0, 120.0, 160.0, 220.0, 300.0]
c_max_target_scan_default = [20.0, 30.0, 40.0, 50.0, 65.0, 80.0]

# Parametri scansione target a due stadi: raffinamento locale dei candidati migliori
top_k_stage1_default = 3
fattori_raffinamento_lambda_default = [0.7, 0.85, 1.0, 1.15, 1.35]
fattori_raffinamento_cmax_default = [0.75, 0.9, 1.0, 1.1, 1.25]

if assetto_prudente_default:
    esegui_scansione_parametri_default = False
    esegui_scansione_trigger_default = False
    esegui_scansione_comportamento_default = False
    esegui_scansione_target_picco_default = False
else:
    esegui_scansione_parametri_default = False
    esegui_scansione_trigger_default = False
    esegui_scansione_comportamento_default = False
    esegui_scansione_target_picco_default = True

T        = 1000        # Durata della simulazione
t        = np.arange(0, T + 1)

# ===== NUOVI PARAMETRI PER STACKELBERG (Cittadini) =====
# Prescrizione sociale: x_bar(c_s)
kappa_prescrizione = 0.05
target_socialita_al_massimo_controllo = 1.0 / (1.0 + kappa_prescrizione * c_max_default)

# Rischio percepito
rho_rischio = 1.5  # scala da I_t/N a rischio individuale

# Utilita cittadino (forma quadratica)
eta_compliance = 12.0  # costo di deviazione dalla prescrizione (compliance)

# Utilita cittadino (forma logaritmica pura): nessun termine quadratico di compliance
a_logaritmica_default = 1.5
epsilon_logaritmica_default = 1e-3
lambda_rischio_logaritmica_default = 4.0
rho_rischio_logaritmica_default = rho_rischio
num_grid_logaritmica_default = 101

# Calibrazione comportamentale utility logaritmica (coarse -> refine)
T_scan_calibrazione_logaritmica_default = 240
a_logaritmica_range_plausibile_default = (0.8, 2.4)
lambda_rischio_logaritmica_range_plausibile_default = (0.6, 2.0)
rho_rischio_range_plausibile_default = (0.8, 2.2)
calibrazione_logaritmica_num_punti_coarse_default = 6  # "a sestante"
calibrazione_logaritmica_num_punti_refine_default = 5
calibrazione_logaritmica_top_k_stage1_default = 4
calibrazione_logaritmica_frazione_refine_default = 0.2

# Numero riproduttivo di base
R0 = beta / gamma
print(f"Numero riproduttivo di base R0 = beta/gamma = {R0:.2f}")
print(f"Condizioni iniziali: S_t0={S_t0}, I_t0={I_t0}, R_t0={R_t0}, N={N}")

# =============================================================================
# 3. Layer follower: comportamento dei cittadini
# =============================================================================
# Questo blocco contiene SOLO le funzioni che descrivono il comportamento
# microeconomico dei cittadini: prescrizione, percezione rischio, utility,
# best response e impatto della socialita sui contatti epidemiologici.

# -----------------------------------------------------------------------------
# 3A) Mappatura policy -> prescrizione e funzioni di utility
# -----------------------------------------------------------------------------

def utilita_cittadino_logaritmica(
    x_t,
    x_bar,
    p_t,
    a,
    epsilon,
    lambda_rischio,
):
    """
    Utility del cittadino (forma LOGARITMICA pura, senza termine quadratico di compliance).

    U_t(x_t) = a * log(x_t/x_bar + epsilon) - lambda * p_t * x_t
    """
    x_t = np.clip(x_t, 0.0, 1.0)
    x_bar = max(x_bar, 1e-8)  # Evita divisione per zero
    # Nuova forma: a * log(x_t/x_bar + epsilon) - lambda * p_t * x_t
    beneficio_socialita = a * np.log(x_t / x_bar + epsilon)
    costo_rischio = lambda_rischio * p_t * x_t
    return beneficio_socialita - costo_rischio


def socialita_prescritta_da_governo(c_s, kappa, tipo="logistica"):
    """
    Mappa/converte la spesa pubblica c_s in un livello di socialita prescritta x_bar(c_s).
    
    Parametri
    ---------
    c_s : float
        Spesa pubblica per controllo.
    kappa : float
        Parametro di sensibilita (quanto velocemente crescono le restrizioni(x_bar diminuisce) al crescere di c_s).
    tipo : str
        "logistica": x_bar = 1 / (1 + kappa * c_s)
        "esponenziale": x_bar = exp(-kappa * c_s)
    
    Ritorna
    -------
    x_bar : float in [0, 1]
        Socialita prescritta dal governo.
    """
    if tipo == "logistica":
        return 1.0 / (1.0 + kappa * max(0, c_s))
    elif tipo == "esponenziale":
        return np.exp(-kappa * max(0, c_s))
    else:
        raise ValueError(f"Tipo prescrizione non riconosciuto: {tipo}")


def rischio_percepito(I_t, N, rho_rischio=1.0):
    """
    Trasforma il numero di infetti in rischio percepito dal cittadino.
    
    Parametri
    ---------
    I_t : float
        Numero di infetti al tempo t.
    N : float
        Popolazione totale.
    rho_rischio : float
        Scala di percezione (default 1.0 => rischio = I_t/N).
    
    Ritorna
    -------
    p_t : float in [0, 1]
        Rischio percepito dal cittadino.
    """
    return np.clip(rho_rischio * (I_t / N), 0.0, 1.0)


# -----------------------------------------------------------------------------
# 3B) Best response dei cittadini (per specifica di utility)
# -----------------------------------------------------------------------------
def best_response_cittadino_quadratica(
    x_bar,
    p_t,
    eta_compliance,
    lambda_rischio=1.0,
):
    """
    Best response del cittadino per utility quadratica.

    La soluzione non vincolata viene poi clippata nell'intervallo [0, 1].
    """
    if eta_compliance > 0:
        x_star_unconstrained = x_bar - (lambda_rischio * p_t) / eta_compliance
    else:
        x_star_unconstrained = x_bar
    x_star = np.clip(x_star_unconstrained, 0.0, 1.0)

    return float(x_star)


def best_response_cittadino_logaritmica(
    x_bar,
    p_t,
    a,
    epsilon,
    lambda_rischio,
    num_grid=101,
):
    """
    Best response del cittadino per utility LOGARITMICA pura (ora dipende da x_bar).

    Massimizza numericamente su griglia [0, 1].
    """
    griglia = np.linspace(0.0, 1.0, num_grid)
    utilita_vals = np.array([
        utilita_cittadino_logaritmica(
            x,
            x_bar,
            p_t,
            a,
            epsilon,
            lambda_rischio,
        )
        for x in griglia
    ])
    idx_best = int(np.argmax(utilita_vals))
    return float(griglia[idx_best])


def fattore_contatto_da_socialita(x_t, potenza=1.0):
    """
    Trasforma socialita effettiva in moltiplicatore del termine di infezione.
    
    # Opzioni:
    # - potenza=1.0 (lineare): contagio proporzionale a x_t.
    # - potenza=2.0 (convesso): contagio proporzionale a x_t^2 (matching).
    
    Parametri
    ---------
    x_t : float in [0, 1]
        Socialita effettiva.
    potenza : float
        Esponente.
    
    Ritorna
    -------
    f : float in [0, 1]
        Fattore moltiplicativo per il termine di contagio.

    # Interpretazione economica/epidemiologica
    # ----------------------------------------
    # Si usa una potenza perche la socialita non deve necessariamente tradursi in
    # contatti epidemiologici in modo lineare. La funzione f(x) = x^potenza permette
    # di scegliere una relazione flessibile:

    # - potenza = 1: relazione lineare, un raddoppio della socialita raddoppia il contatto.
    # - potenza > 1: relazione convessa, utile se i contatti crescono piu che
    #     proporzionalmente quando la socialita si concentra in reti dense o luoghi affollati.
    # - potenza < 1: relazione concava, utile se il numero di contatti cresce meno che
    #     proporzionalmente rispetto alla socialita dichiarata.

    # Nel file corrente viene usato potenza = 1.0, quindi il modello implementato e lineare.
    """
    x_t = np.clip(x_t, 0.0, 1.0)
    return float(x_t ** potenza)


# -----------------------------------------------------------------------------
# 3C) Funzioni di supporto numerico riusabili
# -----------------------------------------------------------------------------
# Queste utility sono trasversali: non definiscono comportamento economico, ma
# supportano stabilita numerica, progress logging e costruzione griglie.
def stampa_progresso_simulazione(
    giorno_corrente,
    giorni_totali,
    label="Simulazione",
    step_percent=step_percent_progress_default,
):
    """Stampa lo stato di avanzamento a step percentuali fissi."""
    if giorni_totali <= 0:
        return

    percentuale = int(np.floor(100 * giorno_corrente / giorni_totali))
    if percentuale >= 100 or percentuale % step_percent == 0:
        print(f"[{label}] progresso: {percentuale:>3}% (giorno {giorno_corrente}/{giorni_totali})")


def somma_sicura(valori):
    """Somma valori finiti saturando al massimo rappresentabile se necessario."""
    totale = np.float64(0.0)
    for valore in valori:
        if not np.isfinite(valore):
            return MAX_FLOAT
        if totale > MAX_FLOAT - valore:
            return MAX_FLOAT
        totale += valore
    return float(totale)


def overload_ospedaliero(H_t, capacita_ospedaliera):
    """Ritorna pressione ospedaliera normalizzata oltre soglia di capacita."""
    capacita_effettiva = max(float(capacita_ospedaliera), 1e-12)
    return max(0.0, (float(H_t) - capacita_effettiva) / capacita_effettiva)


def ricostruisci_comparto_ospedalizzati_da_suscettibili(
    S,
    h_ospedalizzazione=h_ospedalizzazione_default,
    tau_ospedalizzazione=tau_ospedalizzazione_default,
    gamma_ospedaliera=gamma_ospedaliera_default,
    H_t0=H_t0_default,
):
    """Ricostruisce H_t da S_t usando nuovi infetti ritardati e uscite esponenziali."""
    if len(S) < 2:
        return np.array([max(0.0, float(H_t0))], dtype=float)

    nuovi_infetti = np.maximum(0.0, S[:-1] - S[1:])
    T_loc = len(nuovi_infetti)
    H = np.zeros(T_loc + 1, dtype=float)
    H[0] = max(0.0, float(H_t0))

    h_eff = float(np.clip(h_ospedalizzazione, 0.0, 1.0))
    tau_eff = int(max(0, tau_ospedalizzazione))
    gamma_h_eff = float(max(0.0, gamma_ospedaliera))

    for giorno in range(T_loc):
        if giorno >= tau_eff:
            nuovi_ricoveri = h_eff * nuovi_infetti[giorno - tau_eff]
        else:
            nuovi_ricoveri = 0.0

        uscite_H = gamma_h_eff * H[giorno]
        H[giorno + 1] = max(0.0, H[giorno] + nuovi_ricoveri - uscite_H)

    return H


def costruisci_griglia_controllo(c_min, c_max, num_punti):
    """Usa una griglia logaritmica quando i bound coprono molti ordini di grandezza."""
    if num_punti < 2:
        return np.array([float(c_min)])
    if c_max <= c_min:
        return np.array([float(c_min), float(c_max)])

    if c_min > 0.0 and (c_max / c_min) >= 100.0:
        griglia = np.geomspace(c_min, c_max, num_punti)
    else:
        griglia = np.linspace(c_min, c_max, num_punti)

    griglia[0] = c_min
    griglia[-1] = c_max
    return np.unique(griglia)


# =============================================================================
# 4. Funzioni di costo epidemiologico
# =============================================================================
# Questo blocco raccoglie il calcolo dei costi istantanei e cumulati, inclusa
# la variante con controllo variabile nel tempo.

def calcola_costo_epidemiologico_istantaneo(
    S_t,
    I_t,
    costo_infetto_giornaliero,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    H_t=None,
    cap_argomento_exp_saturazione=cap_argomento_exp_saturazione_default,
):
    """Costo epidemiologico istantaneo."""
    _ = S_t
    pressione_ospedaliera = I_t if H_t is None else H_t
    capacita_effettiva = max(float(capacita_ospedaliera_infetti), 1e-12)
    rapporto_occupazione_t = max(0.0, float(pressione_ospedaliera) / capacita_effettiva)
    arg_exp = float(np.clip(alpha_saturazione_ospedali * rapporto_occupazione_t, a_min=0.0, a_max=cap_argomento_exp_saturazione))
    return (
        costo_infetto_giornaliero * I_t
        + k_saturazione_ospedali * (np.exp(arg_exp) - 1.0)
    )


def calcola_costo_epidemiologico_cumulato(
    S,
    I,
    costo_infetto_giornaliero,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    t0,
    giorno_corrente,
    H=None,
    h_ospedalizzazione=h_ospedalizzazione_default,
    tau_ospedalizzazione=tau_ospedalizzazione_default,
    gamma_ospedaliera=gamma_ospedaliera_default,
    H_t0=H_t0_default,
    cap_argomento_exp_saturazione=cap_argomento_exp_saturazione_default,
):
    """Costo epidemiologico cumulato tra t0 e giorno_corrente."""
    if t0 < 0 or giorno_corrente < t0:
        raise ValueError("Indici di tempo non validi.")
    if giorno_corrente >= len(S) or giorno_corrente >= len(I):
        raise ValueError("giorno_corrente eccede le traiettorie.")
    if H is None:
        H = ricostruisci_comparto_ospedalizzati_da_suscettibili(
            S,
            h_ospedalizzazione=h_ospedalizzazione,
            tau_ospedalizzazione=tau_ospedalizzazione,
            gamma_ospedaliera=gamma_ospedaliera,
            H_t0=H_t0,
        )
    if len(H) != len(S):
        raise ValueError("La traiettoria H deve avere la stessa lunghezza di S.")

    costi_giornalieri = [
        calcola_costo_epidemiologico_istantaneo(
            S[giorno],
            I[giorno],
            costo_infetto_giornaliero,
            k_saturazione_ospedali,
            alpha_saturazione_ospedali,
            H_t=H[giorno],
            cap_argomento_exp_saturazione=cap_argomento_exp_saturazione,
        )
        for giorno in range(t0, giorno_corrente + 1)
    ]

    return somma_sicura(costi_giornalieri)


def calcola_traiettoria_costo_epidemiologico_istantaneo(
    S,
    I,
    costo_infetto_giornaliero,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    H=None,
    cap_argomento_exp_saturazione=cap_argomento_exp_saturazione_default,
):
    """Ritorna il costo epidemiologico istantaneo giorno per giorno."""
    if len(S) != len(I):
        raise ValueError("Le traiettorie S e I devono avere la stessa lunghezza.")
    if H is not None and len(H) != len(S):
        raise ValueError("La traiettoria H deve avere la stessa lunghezza di S e I.")

    costi = np.array([
        calcola_costo_epidemiologico_istantaneo(
            S[giorno],
            I[giorno],
            costo_infetto_giornaliero,
            k_saturazione_ospedali,
            alpha_saturazione_ospedali,
            H_t=None if H is None else H[giorno],
            cap_argomento_exp_saturazione=cap_argomento_exp_saturazione,
        )
        for giorno in range(len(S))
    ], dtype=float)
    return costi


def calcola_costo_epidemiologico_cumulato_con_controllo_variabile(
    S,
    I,
    costo_infetto_giornaliero,
    c_schedule,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    t0,
    giorno_corrente,
    lambda_reg_controllo=0.0,
    H=None,
    h_ospedalizzazione=h_ospedalizzazione_default,
    tau_ospedalizzazione=tau_ospedalizzazione_default,
    gamma_ospedaliera=gamma_ospedaliera_default,
    H_t0=H_t0_default,
    cap_argomento_exp_saturazione=cap_argomento_exp_saturazione_default,
):
    """Costo cumulato con c_s variabile nel tempo."""
    if t0 < 0 or giorno_corrente < t0:
        raise ValueError("Indici di tempo non validi.")
    if giorno_corrente >= len(S) or giorno_corrente >= len(I):
        raise ValueError("giorno_corrente eccede le traiettorie.")
    if len(c_schedule) == 0:
        raise ValueError("c_schedule non può essere vuoto.")
    if H is None:
        H = ricostruisci_comparto_ospedalizzati_da_suscettibili(
            S,
            h_ospedalizzazione=h_ospedalizzazione,
            tau_ospedalizzazione=tau_ospedalizzazione,
            gamma_ospedaliera=gamma_ospedaliera,
            H_t0=H_t0,
        )
    if len(H) != len(S):
        raise ValueError("La traiettoria H deve avere la stessa lunghezza di S.")

    costi_giornalieri = [
        (
            calcola_costo_epidemiologico_istantaneo(
                S[giorno],
                I[giorno],
                costo_infetto_giornaliero,
                k_saturazione_ospedali,
                alpha_saturazione_ospedali,
                H_t=H[giorno],
                cap_argomento_exp_saturazione=cap_argomento_exp_saturazione,
            )
            + lambda_reg_controllo * c_schedule[min(giorno, len(c_schedule) - 1)] ** 2
        )
        for giorno in range(t0, giorno_corrente + 1)
    ]

    return somma_sicura(costi_giornalieri)


# =============================================================================
# 5. Dinamica Stackelberg e ottimizzazione del leader
# =============================================================================
# Qui sono definite: simulazione su finestra, costo previsto su finestra,
# ottimizzazione di c_s e simulazione completa con MPC periodico.

# -----------------------------------------------------------------------------
# 5A) Simulazione e costo su finestra di predizione
# -----------------------------------------------------------------------------

def simula_finestra_predizione_stackelberg(
    S_init, I_init, R_init, N, beta, gamma,
    orizzonte, c_s,
    kappa_prescrizione, rho_rischio, eta_compliance,
    a_logaritmica=1.5,
    epsilon_logaritmica=1e-3,
    lambda_rischio_logaritmica=1.0,
    num_grid_logaritmica=101,
    h_ospedalizzazione=h_ospedalizzazione_default,
    tau_ospedalizzazione=tau_ospedalizzazione_default,
    gamma_ospedaliera=gamma_ospedaliera_default,
    H_init=H_t0_default,
    tipo_best_response="quadratica",
):
    """
    Simula una finestra di predizione con decisione cittadino (Stackelberg).
    
    Il governo mantiene c_s costante.
    Ogni giorno il cittadino sceglie x_t^* in risposta a I_t e c_s.
    L'infezione evolve in base a x_t^*.
    
    Parametri
    ---------
    S_init, I_init, R_init : condizioni iniziali
    N : popolazione
    beta, gamma : parametri epidemiologici
    orizzonte : lunghezza finestra
    c_s : spesa pubblica (scelta governo, costante nella finestra)
    kappa_prescrizione : sensibilita prescrizione a c_s
    rho_rischio : scala rischio percepito
    eta_compliance : parametro utility quadratica
    a_logaritmica, epsilon_logaritmica, lambda_rischio_logaritmica, num_grid_logaritmica :
        parametri utility logaritmica pura
    tipo_best_response : "quadratica" oppure "logaritmica"
    
    Ritorna
    -------
    S_pred, I_pred, R_pred, H_pred : traiettorie
    x_bar_pred, x_star_pred : prescrizione e best response
    """
    S_pred = np.zeros(orizzonte + 1)
    I_pred = np.zeros(orizzonte + 1)
    R_pred = np.zeros(orizzonte + 1)
    H_pred = np.zeros(orizzonte + 1)
    x_bar_pred = np.zeros(orizzonte + 1)
    x_star_pred = np.zeros(orizzonte + 1)
    nuovi_infetti_hist = np.zeros(orizzonte)
    
    S_pred[0] = S_init
    I_pred[0] = I_init
    R_pred[0] = R_init
    H_pred[0] = max(0.0, float(H_init))

    h_eff = float(np.clip(h_ospedalizzazione, 0.0, 1.0))
    tau_eff = int(max(0, tau_ospedalizzazione))
    gamma_h_eff = float(max(0.0, gamma_ospedaliera))

    x_bar = socialita_prescritta_da_governo(c_s, kappa_prescrizione, tipo="logistica")
    x_bar_pred[0] = x_bar
    
    for k in range(orizzonte):
        p_k = rischio_percepito(I_pred[k] + H_pred[k], N, rho_rischio)

        if tipo_best_response == "quadratica":
            x_star = best_response_cittadino_quadratica(
                x_bar, p_k, eta_compliance,
            )
        elif tipo_best_response == "logaritmica":
            x_star = best_response_cittadino_logaritmica(
                x_bar, p_k, a_logaritmica, epsilon_logaritmica,
                lambda_rischio_logaritmica, num_grid=num_grid_logaritmica,
            )
        else:
            raise ValueError(f"tipo_best_response non supportato: {tipo_best_response}")
        
        x_star_pred[k] = x_star
        
        fattore = fattore_contatto_da_socialita(x_star, potenza=1.0)
        new_infections = fattore * beta * S_pred[k] * I_pred[k] / N
        nuovi_infetti_hist[k] = new_infections

        if k >= tau_eff:
            new_hospitalizations = h_eff * nuovi_infetti_hist[k - tau_eff]
        else:
            new_hospitalizations = 0.0

        new_recoveries_non_h = gamma * I_pred[k]
        new_recoveries_h = gamma_h_eff * H_pred[k]
        
        S_pred[k + 1] = S_pred[k] - new_infections
        I_pred[k + 1] = max(0.0, I_pred[k] + new_infections - new_hospitalizations - new_recoveries_non_h)
        H_pred[k + 1] = max(0.0, H_pred[k] + new_hospitalizations - new_recoveries_h)
        R_pred[k + 1] = max(0.0, R_pred[k] + new_recoveries_non_h + new_recoveries_h)
        
        x_bar_pred[k + 1] = x_bar
        x_star_pred[k + 1] = x_star
    
    return S_pred, I_pred, R_pred, H_pred, x_bar_pred, x_star_pred


def costo_previsto_su_finestra_stackelberg(
    S_init, I_init, R_init, N, beta, gamma,
    orizzonte, costo_infetto_giornaliero,
    k_saturazione_ospedali, alpha_saturazione_ospedali,
    c_s, kappa_prescrizione,
    rho_rischio,
    eta_compliance, a_logaritmica=1.5, epsilon_logaritmica=1e-3, lambda_rischio_logaritmica=1.0,
    num_grid_logaritmica=101,
    h_ospedalizzazione=h_ospedalizzazione_default,
    tau_ospedalizzazione=tau_ospedalizzazione_default,
    gamma_ospedaliera=gamma_ospedaliera_default,
    H_init=H_t0_default,
    cap_argomento_exp_saturazione=cap_argomento_exp_saturazione_default,
    lambda_reg_controllo=lambda_reg_controllo,
    tipo_best_response="quadratica",
):
    """
    Calcola il costo previsto su una finestra considerando la best response cittadino.

    Il costo totale e dato da due componenti:

        J(c_s) = J_epidemico(c_s) + J_controllo(c_s)

    dove:

        J_epidemico(c_s) = somma dei costi di stato lungo la traiettoria predetta
        J_controllo(c_s) = lambda_reg_controllo * c_s^2 * (orizzonte + 1)

    Nella versione corrente, c_s entra nel costo di stato in modo indiretto,
    perche cambia la traiettoria di I_t tramite la catena:

        c_s -> x_bar(c_s) -> x_t^* -> new_infections -> I_t
    """
    S_pred, I_pred, _, H_pred, _, _ = simula_finestra_predizione_stackelberg(
        S_init, I_init, R_init, N, beta, gamma,
        orizzonte, c_s,
        kappa_prescrizione,
        rho_rischio, eta_compliance,
        a_logaritmica, epsilon_logaritmica, lambda_rischio_logaritmica,
        num_grid_logaritmica,
        h_ospedalizzazione,
        tau_ospedalizzazione,
        gamma_ospedaliera,
        H_init,
        tipo_best_response,
    )
    
    costo_epidemico = calcola_costo_epidemiologico_cumulato(
        S_pred, I_pred,
        costo_infetto_giornaliero, k_saturazione_ospedali, alpha_saturazione_ospedali,
        t0=0, giorno_corrente=orizzonte,
        H=H_pred,
        cap_argomento_exp_saturazione=cap_argomento_exp_saturazione,
    )
    
    costo_controllo = lambda_reg_controllo * c_s ** 2 * (orizzonte + 1)
    return costo_epidemico + costo_controllo


def ottimizza_c_s_su_finestra_stackelberg(
    S_init, I_init, R_init, N, beta, gamma,
    orizzonte, costo_infetto_giornaliero, k_saturazione_ospedali, alpha_saturazione_ospedali,
    c_iniziale, c_min, c_max,
    kappa_prescrizione, rho_rischio,
    eta_compliance, a_logaritmica=1.5, epsilon_logaritmica=1e-3, lambda_rischio_logaritmica=1.0,
    num_grid_logaritmica=101,
    h_ospedalizzazione=h_ospedalizzazione_default,
    tau_ospedalizzazione=tau_ospedalizzazione_default,
    gamma_ospedaliera=gamma_ospedaliera_default,
    H_init=H_t0_default,
    cap_argomento_exp_saturazione=cap_argomento_exp_saturazione_default,
    lambda_reg_controllo=lambda_reg_controllo,
    tipo_best_response="quadratica",
):
    """
    Ottimizza c_s su una finestra con grid search, considerando la reazione cittadino.

    Per ogni candidato c_s nella griglia [c_min, c_max], il governo:
    1. traduce c_s in prescrizione x_bar(c_s);
    2. simula la traiettoria epidemiologica prevista sulla finestra;
    3. calcola il costo totale J(c_s);
    4. sceglie il c_s che minimizza J.

    Questo e un problema di Stackelberg perche il leader (governo) anticipa la
    best response del follower (cittadini) invece di trattare la socialita come esogena.
    """
    c_iniziale_clip = float(np.clip(c_iniziale, c_min, c_max))
    J_iniziale = costo_previsto_su_finestra_stackelberg(
        S_init, I_init, R_init, N, beta, gamma, orizzonte,
        costo_infetto_giornaliero, k_saturazione_ospedali, alpha_saturazione_ospedali,
        c_iniziale_clip, kappa_prescrizione, rho_rischio,
        eta_compliance, a_logaritmica, epsilon_logaritmica, lambda_rischio_logaritmica,
        num_grid_logaritmica,
        h_ospedalizzazione,
        tau_ospedalizzazione,
        gamma_ospedaliera,
        H_init,
        cap_argomento_exp_saturazione,
        lambda_reg_controllo,
        tipo_best_response,
    )
    
    num_grid_points = num_grid_points_default
    griglia_c = costruisci_griglia_controllo(c_min, c_max, num_grid_points)
    valori_J = np.array([
        costo_previsto_su_finestra_stackelberg(
            S_init, I_init, R_init, N, beta, gamma, orizzonte,
            costo_infetto_giornaliero, k_saturazione_ospedali, alpha_saturazione_ospedali,
            c_val, kappa_prescrizione, rho_rischio,
            eta_compliance, 
            a_logaritmica, epsilon_logaritmica, lambda_rischio_logaritmica,
            num_grid_logaritmica,
            h_ospedalizzazione,
            tau_ospedalizzazione,
            gamma_ospedaliera,
            H_init,
            cap_argomento_exp_saturazione,
            lambda_reg_controllo,
            tipo_best_response,
        )
        for c_val in griglia_c
    ])
    
    mask_finiti = np.isfinite(valori_J)
    if not np.any(mask_finiti):
        c_corrente = c_iniziale_clip
        J_corrente = J_iniziale
        stop_reason = "all_nan"
        iter_done = 1
    else:
        idx_validi = np.where(mask_finiti)[0]
        idx_best_rel = int(np.argmin(valori_J[idx_validi]))
        idx_best = int(idx_validi[idx_best_rel])
        c_corrente = float(griglia_c[idx_best])
        J_corrente = float(valori_J[idx_best])
        stop_reason = "grid_search"
        iter_done = len(idx_validi)
    
    if np.isfinite(J_corrente) and np.isfinite(J_iniziale):
        last_delta_costo_assoluto = abs(J_corrente - J_iniziale)
        last_delta_costo_relativo = last_delta_costo_assoluto / (abs(J_iniziale) + 1e-12)
    else:
        last_delta_costo_assoluto = np.inf
        last_delta_costo_relativo = np.inf
    
    info = {
        "iterations": iter_done,
        "stop_reason": stop_reason,
        "last_delta_costo_assoluto": float(last_delta_costo_assoluto),
        "last_delta_costo_relativo": float(last_delta_costo_relativo),
        "c_ottimo": float(c_corrente),
        "J_ottimo": float(J_corrente),
    }
    return c_corrente, info


# -----------------------------------------------------------------------------
# 5B) Simulazione completa con controllo periodico (MPC)
# -----------------------------------------------------------------------------
def simula_sir_stackelberg_con_controllo_periodico(
    S_t0,
    I_t0,
    R_t0,
    N,
    beta,
    gamma,
    T,
    costo_infetto_giornaliero,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    c_iniziale,
    intervallo_controllo=intervallo_controllo_default,
    orizzonte_predizione=orizzonte_predizione_default,
    c_min=c_min_default,
    c_max=c_max_default,
    soglia_attivazione_controllo=soglia_attivazione_controllo_default,
    fattore_isteresi=fattore_isteresi_default,
    lambda_reg_controllo=lambda_reg_controllo,
    kappa_prescrizione=kappa_prescrizione,
    rho_rischio=rho_rischio,
    eta_compliance=eta_compliance,
    a_logaritmica=1.5,
    epsilon_logaritmica=1e-3,
    lambda_rischio_logaritmica=1.0,
    num_grid_logaritmica=101,
    h_ospedalizzazione=h_ospedalizzazione_default,
    tau_ospedalizzazione=tau_ospedalizzazione_default,
    gamma_ospedaliera=gamma_ospedaliera_default,
    H_init=H_t0_default,
    cap_argomento_exp_saturazione=cap_argomento_exp_saturazione_default,
    tipo_best_response="quadratica",
    verbose_progress=verbose_progress_default,
):
    """
    Simula il modello Stackelberg con MPC governo e best response cittadino periodico.
    
    Regola operativa:
    1) Governo decide c_s periodicamente (MPC).
    2) Cittadini osservano c_s e lo stato I_t, scelgono x_t^*.
    3) Dinamica evolve in base a x_t^*.
    """
    S = np.zeros(T + 1)
    I = np.zeros(T + 1)
    R = np.zeros(T + 1)
    H = np.zeros(T + 1)
    nuovi_infetti_hist = np.zeros(T)
    S[0], I[0], R[0] = S_t0, I_t0, R_t0
    H[0] = max(0.0, float(H_init))
    
    c_schedule = np.zeros(T)
    x_bar_schedule = np.zeros(T)
    x_star_schedule = np.zeros(T)
    p_rischio_schedule = np.zeros(T)
    log_ottimizzazione = []
    
    giorno = 0
    c_guess = float(np.clip(c_iniziale, c_min, c_max))
    ultimo_checkpoint_stampato = -1
    controllo_attivo = False
    prossimo_giorno_ottimizzazione = None
    c_applicato = c_min
    soglia_infetti_assoluta = soglia_attivazione_controllo * N
    soglia_bassa = soglia_infetti_assoluta * fattore_isteresi
    h_eff = float(np.clip(h_ospedalizzazione, 0.0, 1.0))
    tau_eff = int(max(0, tau_ospedalizzazione))
    gamma_h_eff = float(max(0.0, gamma_ospedaliera))

    if tipo_best_response not in ("quadratica", "logaritmica"):
        raise ValueError(
            "tipo_best_response deve essere 'quadratica' oppure 'logaritmica'."
        )
    
    if verbose_progress:
        print("\n[Stackelberg MPC] avvio simulazione...")
        stampa_progresso_simulazione(0, T, label="Stackelberg MPC")
    
    while giorno < T:
        # Attivazione/disattivazione controllo su soglia infetti
        infetti_attivi = I[giorno] + H[giorno]
        if (not controllo_attivo) and (infetti_attivi >= soglia_infetti_assoluta):
            controllo_attivo = True
            prossimo_giorno_ottimizzazione = giorno
        
        if controllo_attivo and (infetti_attivi < soglia_bassa):
            controllo_attivo = False
            prossimo_giorno_ottimizzazione = None
            c_applicato = c_min
        
        # Ottimizzazione periodica del governo
        if controllo_attivo and (prossimo_giorno_ottimizzazione is not None) and (giorno >= prossimo_giorno_ottimizzazione):
            orizzonte_ottimizzazione = min(orizzonte_predizione, T - giorno)
            
            c_ottimo, info = ottimizza_c_s_su_finestra_stackelberg(
                S[giorno], I[giorno],  R[giorno], N, beta, gamma,
                orizzonte_ottimizzazione, costo_infetto_giornaliero,
                k_saturazione_ospedali, alpha_saturazione_ospedali,
                c_guess, c_min, c_max,
                kappa_prescrizione, rho_rischio, eta_compliance,
                a_logaritmica, epsilon_logaritmica, lambda_rischio_logaritmica,
                num_grid_logaritmica,
                h_eff,
                tau_eff,
                gamma_h_eff,
                H[giorno],
                cap_argomento_exp_saturazione,
                lambda_reg_controllo,
                tipo_best_response,
            )
            
            c_applicato = c_ottimo
            info["giorno_start"] = int(giorno)
            info["giorno_end"] = int(min(giorno + intervallo_controllo, T))
            info["orizzonte_predizione"] = int(orizzonte_ottimizzazione)
            log_ottimizzazione.append(info)
            
            c_guess = c_ottimo
            prossimo_giorno_ottimizzazione = giorno + intervallo_controllo
        elif not controllo_attivo:
            c_applicato = c_min
        
        # Decisione cittadino e evoluzione dinamica
        x_bar = socialita_prescritta_da_governo(c_applicato, kappa_prescrizione, tipo="logistica")
        p_t = rischio_percepito(infetti_attivi, N, rho_rischio)

        if tipo_best_response == "quadratica":
            x_star = best_response_cittadino_quadratica(
                x_bar,
                p_t,
                eta_compliance,
            )
        else:
            x_star = best_response_cittadino_logaritmica(
                x_bar,
                p_t,
                a_logaritmica,
                epsilon_logaritmica,
                lambda_rischio_logaritmica,
                num_grid=num_grid_logaritmica,
            )
        
        fattore = fattore_contatto_da_socialita(x_star, potenza=1.0)
        new_infections = fattore * beta * S[giorno] * I[giorno] / N
        nuovi_infetti_hist[giorno] = new_infections

        if giorno >= tau_eff:
            new_hospitalizations = h_eff * nuovi_infetti_hist[giorno - tau_eff]
        else:
            new_hospitalizations = 0.0

        new_recoveries_non_h = gamma * I[giorno]
        new_recoveries_h = gamma_h_eff * H[giorno]
        
        S[giorno + 1] = S[giorno] - new_infections
        I[giorno + 1] = max(0.0, I[giorno] + new_infections - new_hospitalizations - new_recoveries_non_h)
        H[giorno + 1] = max(0.0, H[giorno] + new_hospitalizations - new_recoveries_h)
        R[giorno + 1] = max(0.0, R[giorno] + new_recoveries_non_h + new_recoveries_h)
        
        c_schedule[giorno] = c_applicato
        x_bar_schedule[giorno] = x_bar
        x_star_schedule[giorno] = x_star
        p_rischio_schedule[giorno] = p_t
        
        if verbose_progress:
            percentuale = int(np.floor(100 * (giorno + 1) / T))
            checkpoint = percentuale // 5
            if checkpoint > ultimo_checkpoint_stampato:
                ultimo_checkpoint_stampato = checkpoint
                stampa_progresso_simulazione(giorno + 1, T, label="Stackelberg MPC")
        
        giorno += 1
    
    if verbose_progress:
        print("[Stackelberg MPC] simulazione completata.")
    
    return S, I, R, c_schedule, x_bar_schedule, x_star_schedule, p_rischio_schedule, log_ottimizzazione


# =============================================================================
# 6. Analisi output: grafici e scansioni di calibrazione
# =============================================================================
# Questo blocco e dedicato a funzioni di post-analisi: plotting e benchmark.

# -----------------------------------------------------------------------------
# 6A) Funzioni di visualizzazione
# -----------------------------------------------------------------------------

def plot_dinamica_compartimenti_stackelberg(
    t, S, I, R, N, beta, gamma, R0, t_picco,
    H=None,
    simulation_label="quadratica",
    output_path="sir_compartimenti_stackelberg.png",
):
    """Visualizza dinamica compartimenti (stessa logica di deterministic.py)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Modello SIR Stackelberg — Governo vs Cittadini\n"
        f"Utility cittadini: {simulation_label}  |  "
        f"$N={N}$,  $\\beta={beta}$,  $\\gamma={gamma}$,  $R_0={R0:.1f}$",
        fontsize=14,
        fontweight="bold",
    )
    
    ax1 = axes[0]
    ax1.plot(t, S / N * 100, color="steelblue", lw=2, label="S — Suscettibili")
    ax1.plot(t, I / N * 100, color="crimson", lw=2, label="I — Infetti (con Stackelberg)")
    ax1.plot(t, R / N * 100, color="forestgreen", lw=2, label="R — Rimossi/Guariti")
    if H is not None:
        ax1.plot(t, H / N * 100, color="darkorange", lw=2, label="H — Ospedalizzati")
    ax1.axvline(t_picco, color="crimson", lw=1.2, ls="--", alpha=0.6,
                label=f"Picco I (giorno {t_picco})")
    ax1.set_ylabel("Frazione di popolazione (%)", fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Evoluzione dei compartimenti nel tempo", fontsize=11)
    
    ax2 = axes[1]
    # Incidenza realizzata coerente con la dinamica simulata.
    nuovi_infetti = np.maximum(0.0, S[:-1] - S[1:])
    ax2.bar(t[:-1], nuovi_infetti, color="crimson", alpha=0.6, width=1,
            label="Nuovi infetti / giorno")
    if H is not None:
        ax2_right = ax2.twinx()
        ax2_right.plot(t, H, color="darkorange", lw=1.8, alpha=0.85, label="Ospedalizzati H(t)")
        ax2_right.set_ylabel("Ospedalizzati", fontsize=11, color="darkorange")
        ax2_right.tick_params(axis="y", labelcolor="darkorange")
        lines_left, labels_left = ax2.get_legend_handles_labels()
        lines_right, labels_right = ax2_right.get_legend_handles_labels()
        ax2.legend(lines_left + lines_right, labels_left + labels_right, fontsize=10, loc="upper right")
    else:
        ax2.legend(fontsize=10)
    ax2.set_xlabel("Tempo (giorni)", fontsize=11)
    ax2.set_ylabel("Nuovi infetti", fontsize=11)
    ax2.set_title("Incidenza giornaliera (flusso S → I)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def plot_controllo_e_comportamento_stackelberg(
    t, S, I, c_schedule, x_bar_schedule, x_star_schedule, p_rischio_schedule,
    H=None,
    simulation_label="quadratica",
    output_path="sir_stackelberg_analisi.png",
):
    """Visualizza c_s, prescrizione e best response nel tempo."""
    t_controllo = t[:-1]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    costo_istantaneo = calcola_traiettoria_costo_epidemiologico_istantaneo(
        S,
        I,
        costo_infetto_giornaliero,
        k_saturazione_ospedali,
        alpha_saturazione_ospedali,
        H=H,
    )
    costo_istantaneo_controllo = costo_istantaneo[:-1]
    
    # c_s(t)
    axes[0].step(t_controllo, c_schedule, where="post", lw=2, color="purple", label="c_s(t) — Governo")
    axes[0].set_ylabel("c_s", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    ax0_right = axes[0].twinx()
    ax0_right.plot(
        t_controllo,
        costo_istantaneo_controllo,
        lw=2,
        color="black",
        alpha=0.85,
        label="Costo epidemico istantaneo",
    )
    ax0_right.set_ylabel("Costo epidemico", fontsize=10, color="black")
    ax0_right.tick_params(axis="y", labelcolor="black")
    lines_left, labels_left = axes[0].get_legend_handles_labels()
    lines_right, labels_right = ax0_right.get_legend_handles_labels()
    axes[0].legend(lines_left + lines_right, labels_left + labels_right, fontsize=9)
    axes[0].set_title("Spesa pubblica e costo epidemico istantaneo", fontsize=11)
    
    # x_bar vs x_star
    axes[1].plot(t_controllo, x_bar_schedule, lw=2, color="orange", label="$\\bar{x}(t)$ — Prescrizione", marker='o', markersize=2)
    axes[1].plot(t_controllo, x_star_schedule, lw=2, color="green", label="$x^*(t)$ — Best response", marker='s', markersize=2)
    axes[1].set_ylabel("Socialita", fontsize=10)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].set_title("Prescrizione vs comportamento cittadini", fontsize=11)
    
    # Compliance gap
    compliance_gap = x_star_schedule - x_bar_schedule
    axes[2].bar(t_controllo, compliance_gap, color=['green' if gap >= 0 else 'red' for gap in compliance_gap],
                alpha=0.6, width=1, label="Compliance gap ($x^* - \\bar{x}$)")
    axes[2].axhline(0, color='black', lw=0.5)
    axes[2].set_ylabel("Gap", fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)
    axes[2].set_title("Scostamento dalla prescrizione", fontsize=11)
    
    # Rischio percepito
    axes[3].plot(t_controllo, p_rischio_schedule, lw=2, color="darkred", label="$p_t$ — Rischio percepito")
    axes[3].fill_between(t_controllo, 0, p_rischio_schedule, alpha=0.3, color="darkred")
    axes[3].set_ylabel("Rischio", fontsize=10)
    axes[3].set_ylim(0, 1.05)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=9)
    axes[3].set_title("Rischio epidemico percepito", fontsize=11)

    axes[3].set_xlabel("Tempo (giorni)", fontsize=11)
    
    fig.suptitle(
        f"Analisi Stackelberg ({simulation_label}): Policy governo vs Comportamento cittadini",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


# -----------------------------------------------------------------------------
# 6B) Scansioni e calibrazioni parametriche
# -----------------------------------------------------------------------------
# Queste funzioni non fanno parte del core dinamico: servono per sensitivity
# analysis, calibrazione e confronto tra scenari.
def esegui_scansione_alpha_lambda(
    alpha_values, lambda_values, T_scan,
):
    """Esegue una scansione su alpha e lambda e stampa tabella di confronto."""
    risultati = []

    for alpha_val in alpha_values:
        for lambda_val in lambda_values:
            _, I_tmp, _, c_tmp, _, _, _, _ = simula_sir_stackelberg_con_controllo_periodico(
                S_t0, I_t0, R_t0, N, beta, gamma,
                T_scan, costo_infetto_giornaliero, k_saturazione_ospedali, alpha_val,
                c_iniziale=c_iniziale_default,
                intervallo_controllo=intervallo_controllo_default,
                orizzonte_predizione=orizzonte_predizione_default,
                c_min=c_min_default, c_max=c_max_default,
                soglia_attivazione_controllo=soglia_attivazione_controllo_default,
                fattore_isteresi=fattore_isteresi_default,
                lambda_reg_controllo=lambda_val,
                kappa_prescrizione=kappa_prescrizione,
                rho_rischio=rho_rischio, eta_compliance=eta_compliance,
                tipo_best_response="quadratica",
                verbose_progress=False,
            )

            t_picco = int(np.argmax(I_tmp))
            picco_infetti = float(np.max(I_tmp))
            quota_tempo_cmax = float(np.mean(np.isclose(c_tmp, c_max_default)))

            risultati.append({
                "alpha": float(alpha_val),
                "lambda": float(lambda_val),
                "picco": picco_infetti,
                "giorno_picco": t_picco,
                "quota_tempo_cmax": quota_tempo_cmax,
            })

    risultati.sort(key=lambda x: (x["picco"], x["quota_cmax"]))

    print("\n" + "=" * 80)
    print("SCAN CALIBRAZIONE PARAMETRI (alpha_saturazione_ospedali, lambda_reg_controllo)")
    print("=" * 80)
    print(f"Orizzonte scansione: T_scan={T_scan} giorni")
    print("alpha | lambda | picco_infetti | giorno_picco | quota_tempo_a_cmax")
    print("-" * 80)
    for riga in risultati:
        print(
            f"{riga['alpha']:>5.1f} | "
            f"{riga['lambda']:>6.1f} | "
            f"{riga['picco']:>13.0f} | "
            f"{riga['giorno_picco']:>11d} | "
            f"{riga['quota_cmax']:>18.3f}"
        )

    return risultati


def esegui_scansione_trigger_isteresi(
    soglia_values, isteresi_values, T_scan,
):
    """Esegue una scansione su soglia di attivazione e isteresi del controllo."""
    risultati = []

    for soglia_val in soglia_values:
        for isteresi_val in isteresi_values:
            _, I_tmp, _, c_tmp, _, _, _, _ = simula_sir_stackelberg_con_controllo_periodico(
                S_t0,
                I_t0,
                R_t0,
                N,
                beta,
                gamma,
                T_scan,
                costo_infetto_giornaliero,
                k_saturazione_ospedali,
                alpha_saturazione_ospedali,
                c_iniziale=c_iniziale_default,
                intervallo_controllo=intervallo_controllo_default,
                orizzonte_predizione=orizzonte_predizione_default,
                c_min=c_min_default,
                c_max=c_max_default,
                soglia_attivazione_controllo=soglia_val,
                fattore_isteresi=isteresi_val,
                lambda_reg_controllo=lambda_reg_controllo,
                kappa_prescrizione=kappa_prescrizione,
                rho_rischio=rho_rischio,
                eta_compliance=eta_compliance,
                tipo_best_response="quadratica",
                verbose_progress=False,
            )

            t_picco = int(np.argmax(I_tmp))
            picco_infetti = float(np.max(I_tmp))
            quota_tempo_cmax = float(np.mean(np.isclose(c_tmp, c_max_default)))
            quota_tempo_controllo_attivo = float(np.mean(c_tmp > c_min_default + 1e-9))

            risultati.append({
                "soglia": float(soglia_val),
                "isteresi": float(isteresi_val),
                "picco": picco_infetti,
                "giorno_picco": t_picco,
                "quota_cmax": quota_tempo_cmax,
                "quota_controllo_attivo": quota_tempo_controllo_attivo,
            })

    risultati.sort(key=lambda x: (x["picco"], x["giorno_picco"]))

    print("\n" + "=" * 80)
    print("SCAN CALIBRAZIONE PARAMETRI (soglia_attivazione_controllo, fattore_isteresi)")
    print("=" * 80)
    print(f"Orizzonte scansione: T_scan={T_scan} giorni")
    print("soglia | isteresi | picco_infetti | giorno_picco | quota_tempo_a_cmax | quota_tempo_controllo_attivo")
    print("-" * 110)
    for riga in risultati:
        print(
            f"{riga['soglia']:>6.4f} | "
            f"{riga['isteresi']:>8.2f} | "
            f"{riga['picco']:>13.0f} | "
            f"{riga['giorno_picco']:>11d} | "
            f"{riga['quota_cmax']:>18.3f} | "
            f"{riga['quota_controllo_attivo']:>26.3f}"
        )

    return risultati


def esegui_scansione_comportamento(
    kappa_values, eta_values, rho_values, T_scan,
):
    """Esegue una scansione su prescrizione, compliance e rischio percepito."""
    risultati = []

    for kappa_val in kappa_values:
        for eta_val in eta_values:
            for rho_val in rho_values:
                _, I_tmp, _, c_tmp, x_bar_tmp, x_star_tmp, _, _ = simula_sir_stackelberg_con_controllo_periodico(
                    S_t0,
                    I_t0,
                    R_t0,
                    N,
                    beta,
                    gamma,
                    T_scan,
                    costo_infetto_giornaliero,
                    k_saturazione_ospedali,
                    alpha_saturazione_ospedali,
                    c_iniziale=c_iniziale_default,
                    intervallo_controllo=intervallo_controllo_default,
                    orizzonte_predizione=orizzonte_predizione_default,
                    c_min=c_min_default,
                    c_max=c_max_default,
                    soglia_attivazione_controllo=soglia_attivazione_controllo_default,
                    fattore_isteresi=fattore_isteresi_default,
                    lambda_reg_controllo=lambda_reg_controllo,
                    kappa_prescrizione=kappa_val,
                    rho_rischio=rho_val,
                    eta_compliance=eta_val,
                    tipo_best_response="quadratica",
                    verbose_progress=False,
                )

                t_picco = int(np.argmax(I_tmp))
                picco_infetti = float(np.max(I_tmp))
                quota_tempo_cmax = float(np.mean(np.isclose(c_tmp, c_max_default)))
                gap_medio = float(np.mean(x_star_tmp - x_bar_tmp))

                risultati.append({
                    "kappa": float(kappa_val),
                    "eta": float(eta_val),
                    "rho": float(rho_val),
                    "picco": picco_infetti,
                    "giorno_picco": t_picco,
                    "quota_cmax": quota_tempo_cmax,
                    "gap_medio": gap_medio,
                })

    risultati.sort(key=lambda x: (x["picco"], x["quota_cmax"], abs(x["gap_medio"])))

    print("\n" + "=" * 96)
    print("SCAN CALIBRAZIONE PARAMETRI (kappa_prescrizione, eta_compliance, rho_rischio)")
    print("=" * 96)
    print(f"Orizzonte scansione: T_scan={T_scan} giorni")
    print("Nota: rho_rischio entra nella utility quadratica tramite p_t = clip(rho * I_t / N, 0, 1).")
    print("kappa | eta | rho | picco_infetti | giorno_picco | quota_tempo_a_cmax | gap_medio")
    print("-" * 96)
    for riga in risultati:
        print(
            f"{riga['kappa']:>5.3f} | "
            f"{riga['eta']:>4.1f} | "
            f"{riga['rho']:>3.2f} | "
            f"{riga['picco']:>13.0f} | "
            f"{riga['giorno_picco']:>11d} | "
            f"{riga['quota_cmax']:>18.3f} | "
            f"{riga['gap_medio']:>9.4f}"
        )

    return risultati


def calibra_parametri_logaritmica_min_picco_due_stadi(
    T_scan,
    a_range,
    lambda_rischio_range,
    rho_range,
    num_punti_coarse=6,
    num_punti_refine=5,
    top_k_stage1=4,
    frazione_refine=0.2,
):
    """
    Calibrazione comportamentale utility logaritmica in due stadi.

    STADIO 1: griglia grossolana "a sestante" nel range plausibile.
    STADIO 2: raffinamento locale attorno ai migliori candidati stage 1.

    Criterio di selezione: minimizzare il picco massimo degli infetti.
    """
    def valuta_tripla(a_val, lambda_rischio_val, rho_val):
        S_tmp, I_tmp, _, c_tmp, x_bar_tmp, x_star_tmp, _, _ = simula_sir_stackelberg_con_controllo_periodico(
            S_t0,
            I_t0,
            R_t0,
            N,
            beta,
            gamma,
            T_scan,
            costo_infetto_giornaliero,
            k_saturazione_ospedali,
            alpha_saturazione_ospedali,
            c_iniziale=c_iniziale_default,
            intervallo_controllo=intervallo_controllo_default,
            orizzonte_predizione=orizzonte_predizione_default,
            c_min=c_min_default,
            c_max=c_max_default,
            soglia_attivazione_controllo=soglia_attivazione_controllo_default,
            fattore_isteresi=fattore_isteresi_default,
            lambda_reg_controllo=lambda_reg_controllo,
            kappa_prescrizione=kappa_prescrizione,
            rho_rischio=rho_val,
            eta_compliance=eta_compliance,
            a_logaritmica=a_val,
            epsilon_logaritmica=epsilon_logaritmica_default,
            lambda_rischio_logaritmica=lambda_rischio_val,
            num_grid_logaritmica=num_grid_logaritmica_default,
            tipo_best_response="logaritmica",
            verbose_progress=False,
        )

        picco_infetti = float(np.max(I_tmp))
        picco_percent = float(100.0 * picco_infetti / N)
        giorno_picco = int(np.argmax(I_tmp))
        quota_tempo_cmax = float(np.mean(np.isclose(c_tmp, c_max_default, rtol=1e-7, atol=1e-9)))
        compliance_media = float(np.mean([x_star_tmp[i] / max(x_bar_tmp[i], 0.01) for i in range(len(x_star_tmp))])) if T_scan > 0 else np.nan
        costo_totale = calcola_costo_epidemiologico_cumulato_con_controllo_variabile(
            S_tmp,
            I_tmp,
            costo_infetto_giornaliero,
            c_tmp,
            k_saturazione_ospedali,
            alpha_saturazione_ospedali,
            0,
            T_scan,
        )

        return {
            "a_logaritmica": float(a_val),
            "lambda_rischio_logaritmica": float(lambda_rischio_val),
            "rho_rischio": float(rho_val),
            "picco": picco_infetti,
            "picco_percent": picco_percent,
            "giorno_picco": giorno_picco,
            "quota_cmax": quota_tempo_cmax,
            "compliance_media": compliance_media,
            "costo_totale": float(costo_totale),
        }

    def sort_key(item):
        return (item["picco"], item["costo_totale"], item["quota_cmax"])

    a_values_stage1 = np.linspace(a_range[0], a_range[1], num_punti_coarse)
    lambda_values_stage1 = np.linspace(lambda_rischio_range[0], lambda_rischio_range[1], num_punti_coarse)
    rho_values_stage1 = np.linspace(rho_range[0], rho_range[1], num_punti_coarse)

    risultati_stage1 = [
        valuta_tripla(a_val, lambda_val, rho_val)
        for a_val in a_values_stage1
        for lambda_val in lambda_values_stage1
        for rho_val in rho_values_stage1
    ]
    risultati_stage1.sort(key=sort_key)

    top_k_eff = max(1, min(top_k_stage1, len(risultati_stage1)))
    semi_stage2 = risultati_stage1[:top_k_eff]

    ampiezza_a = max((a_range[1] - a_range[0]) * frazione_refine, 1e-8)
    ampiezza_lambda = max((lambda_rischio_range[1] - lambda_rischio_range[0]) * frazione_refine, 1e-8)
    ampiezza_rho = max((rho_range[1] - rho_range[0]) * frazione_refine, 1e-8)

    triple_stage2 = set()
    for seed in semi_stage2:
        a_local_min = max(a_range[0], seed["a_logaritmica"] - ampiezza_a)
        a_local_max = min(a_range[1], seed["a_logaritmica"] + ampiezza_a)
        lambda_local_min = max(lambda_rischio_range[0], seed["lambda_rischio_logaritmica"] - ampiezza_lambda)
        lambda_local_max = min(lambda_rischio_range[1], seed["lambda_rischio_logaritmica"] + ampiezza_lambda)
        rho_local_min = max(rho_range[0], seed["rho_rischio"] - ampiezza_rho)
        rho_local_max = min(rho_range[1], seed["rho_rischio"] + ampiezza_rho)

        a_values_local = np.linspace(a_local_min, a_local_max, num_punti_refine)
        lambda_values_local = np.linspace(lambda_local_min, lambda_local_max, num_punti_refine)
        rho_values_local = np.linspace(rho_local_min, rho_local_max, num_punti_refine)

        for a_val in a_values_local:
            for lambda_val in lambda_values_local:
                for rho_val in rho_values_local:
                    triple_stage2.add((round(float(a_val), 10), round(float(lambda_val), 10), round(float(rho_val), 10)))

    risultati_stage2 = [
        valuta_tripla(a_val, lambda_val, rho_val)
        for (a_val, lambda_val, rho_val) in sorted(triple_stage2)
    ]
    risultati_stage2.sort(key=sort_key)
    migliore = risultati_stage2[0]

    print("\n" + "=" * 126)
    print("CALIBRAZIONE UTILITY LOGARITMICA PURA (MINIMIZZAZIONE PICCO) - DUE STADI")
    print("=" * 126)
    print(
        f"Range plausibili: a in [{a_range[0]:.2f}, {a_range[1]:.2f}], "
        f"lambda_rischio in [{lambda_rischio_range[0]:.2f}, {lambda_rischio_range[1]:.2f}], "
        f"rho in [{rho_range[0]:.2f}, {rho_range[1]:.2f}]"
    )
    print(
        f"Stadio 1 (coarse): {num_punti_coarse}x{num_punti_coarse}x{num_punti_coarse} = {len(risultati_stage1)} scenari | "
        f"top_k={top_k_eff}"
    )
    print(f"Stadio 2 (refine): {len(risultati_stage2)} scenari | orizzonte calibrazione T_scan={T_scan} giorni")

    print("\nTop scenari STADIO 1 (primi 10):")
    print("a_log | lambda_rischio | rho | picco_infetti | picco_% | giorno_picco | costo_totale")
    print("-" * 108)
    for riga in risultati_stage1[:10]:
        print(
            f"{riga['a_logaritmica']:>5.2f} | "
            f"{riga['lambda_rischio_logaritmica']:>14.2f} | "
            f"{riga['rho_rischio']:>3.2f} | "
            f"{riga['picco']:>13.0f} | "
            f"{riga['picco_percent']:>7.2f} | "
            f"{riga['giorno_picco']:>11d} | "
            f"{riga['costo_totale']:>11,.0f}"
        )

    print("\nTop scenari STADIO 2 (primi 12):")
    print("a_log | lambda_rischio | rho | picco_infetti | picco_% | giorno_picco | costo_totale")
    print("-" * 108)
    for riga in risultati_stage2[:12]:
        print(
            f"{riga['a_logaritmica']:>5.2f} | "
            f"{riga['lambda_rischio_logaritmica']:>14.2f} | "
            f"{riga['rho_rischio']:>3.2f} | "
            f"{riga['picco']:>13.0f} | "
            f"{riga['picco_percent']:>7.2f} | "
            f"{riga['giorno_picco']:>11d} | "
            f"{riga['costo_totale']:>11,.0f}"
        )

    print("\nMiglior scenario finale (picco minimo):")
    print(
        f"  a_logaritmica={migliore['a_logaritmica']:.3f}, "
        f"lambda_rischio_logaritmica={migliore['lambda_rischio_logaritmica']:.3f}, "
        f"rho_rischio={migliore['rho_rischio']:.3f}, "
        f"picco={migliore['picco']:.0f} ({migliore['picco_percent']:.2f}%), "
        f"giorno_picco={migliore['giorno_picco']}, costo_totale={migliore['costo_totale']:,.0f}"
    )

    return {
        "stage1": risultati_stage1,
        "stage2": risultati_stage2,
        "migliore": migliore,
    }


def valuta_scenario_target_picco(
    lambda_val,
    c_max_val,
    T_scan,
    target_picco_percent,
    tolleranza_percent,
):
    """Valuta un singolo scenario (lambda, c_max) rispetto al target di picco."""
    c_min_eff = min(c_min_default, c_max_val)
    c_init_eff = float(np.clip(c_iniziale_default, c_min_eff, c_max_val))

    S_tmp, I_tmp, _, c_tmp, _, _, _, _ = simula_sir_stackelberg_con_controllo_periodico(
        S_t0,
        I_t0,
        R_t0,
        N,
        beta,
        gamma,
        T_scan,
        costo_infetto_giornaliero,
        k_saturazione_ospedali,
        alpha_saturazione_ospedali,
        c_iniziale=c_init_eff,
        intervallo_controllo=intervallo_controllo_default,
        orizzonte_predizione=orizzonte_predizione_default,
        c_min=c_min_eff,
        c_max=c_max_val,
        soglia_attivazione_controllo=soglia_attivazione_controllo_default,
        fattore_isteresi=fattore_isteresi_default,
        lambda_reg_controllo=lambda_val,
        kappa_prescrizione=kappa_prescrizione,
        rho_rischio=rho_rischio,
        eta_compliance=eta_compliance,
        tipo_best_response="quadratica",
        verbose_progress=False,
    )

    picco_infetti = float(np.max(I_tmp))
    picco_percent = 100.0 * picco_infetti / N
    t_picco = int(np.argmax(I_tmp))
    distanza_target_percent = abs(picco_percent - target_picco_percent)
    in_tolleranza = distanza_target_percent <= tolleranza_percent
    quota_tempo_cmax = float(np.mean(np.isclose(c_tmp, c_max_val, rtol=1e-7, atol=1e-9)))

    costo_totale = calcola_costo_epidemiologico_cumulato_con_controllo_variabile(
        S_tmp,
        I_tmp,
        costo_infetto_giornaliero,
        c_tmp,
        k_saturazione_ospedali,
        alpha_saturazione_ospedali,
        0,
        T_scan,
        lambda_val,
    )

    return {
        "lambda": float(lambda_val),
        "c_max": float(c_max_val),
        "picco": picco_infetti,
        "picco_percent": float(picco_percent),
        "distanza_target_percent": float(distanza_target_percent),
        "in_tolleranza": bool(in_tolleranza),
        "giorno_picco": t_picco,
        "quota_cmax": quota_tempo_cmax,
        "costo_totale": float(costo_totale),
    }


def esegui_scansione_target_picco(
    lambda_values, c_max_values, T_scan,
    target_picco_percent=10.0, tolleranza_percent=1.0,
):
    """
    Esegue una scansione su (lambda_reg_controllo, c_max) con benchmark sul picco infetti.

    # Obiettivo principale: avvicinare il picco infetti a target_picco_percent della popolazione.
    # L'ordinamento finale usa come priorita:
    # 1) distanza assoluta dal target;
    # 2) minor quota di tempo a c_max (evita soluzioni sempre in saturazione);
    # 3) minor costo totale.
    """
    risultati = []
    target_assoluto = (target_picco_percent / 100.0) * N

    for lambda_val in lambda_values:
        for c_max_val in c_max_values:
            c_min_eff = min(c_min_default, c_max_val)
            c_init_eff = float(np.clip(c_iniziale_default, c_min_eff, c_max_val))

            S_tmp, I_tmp, _, c_tmp, _, _, _, _ = simula_sir_stackelberg_con_controllo_periodico(
                S_t0,
                I_t0,
                R_t0,
                N,
                beta,
                gamma,
                T_scan,
                costo_infetto_giornaliero,
                k_saturazione_ospedali,
                alpha_saturazione_ospedali,
                c_iniziale=c_init_eff,
                intervallo_controllo=intervallo_controllo_default,
                orizzonte_predizione=orizzonte_predizione_default,
                c_min=c_min_eff,
                c_max=c_max_val,
                soglia_attivazione_controllo=soglia_attivazione_controllo_default,
                fattore_isteresi=fattore_isteresi_default,
                lambda_reg_controllo=lambda_val,
                kappa_prescrizione=kappa_prescrizione,
                rho_rischio=rho_rischio,
                eta_compliance=eta_compliance,
                tipo_best_response="quadratica",
                verbose_progress=False,
            )

            picco_infetti = float(np.max(I_tmp))
            picco_percent = 100.0 * picco_infetti / N
            t_picco = int(np.argmax(I_tmp))
            distanza_target_percent = abs(picco_percent - target_picco_percent)
            in_tolleranza = distanza_target_percent <= tolleranza_percent
            quota_tempo_cmax = float(np.mean(np.isclose(c_tmp, c_max_val, rtol=1e-7, atol=1e-9)))

            costo_totale = calcola_costo_epidemiologico_cumulato_con_controllo_variabile(
                S_tmp,
                I_tmp,
                costo_infetto_giornaliero,
                c_tmp,
                k_saturazione_ospedali,
                alpha_saturazione_ospedali,
                0,
                T_scan,
                lambda_val,
            )

            risultati.append({
                "lambda": float(lambda_val),
                "c_max": float(c_max_val),
                "picco": picco_infetti,
                "picco_percent": float(picco_percent),
                "target_assoluto": float(target_assoluto),
                "distanza_target_percent": float(distanza_target_percent),
                "in_tolleranza": bool(in_tolleranza),
                "giorno_picco": t_picco,
                "quota_cmax": quota_tempo_cmax,
                "costo_totale": float(costo_totale),
            })

    risultati.sort(
        key=lambda x: (
            x["distanza_target_percent"],
            x["quota_cmax"],
            x["costo_totale"],
        )
    )

    print("\n" + "=" * 112)
    print("SCAN CALIBRAZIONE TARGET PICCO (lambda_reg_controllo, c_max)")
    print("=" * 112)
    print(
        f"Target picco: {target_picco_percent:.1f}% della popolazione "
        f"(~{target_assoluto:.0f} infetti), tolleranza +/-{tolleranza_percent:.1f}%"
    )
    print(f"Orizzonte scansione: T_scan={T_scan} giorni")
    print("lambda | c_max | picco_infetti | picco_% | dist_target_% | in_tol | giorno_picco | quota_tempo_a_cmax")
    print("-" * 112)

    for riga in risultati:
        print(
            f"{riga['lambda']:>6.1f} | "
            f"{riga['c_max']:>5.1f} | "
            f"{riga['picco']:>13.0f} | "
            f"{riga['picco_percent']:>7.2f} | "
            f"{riga['distanza_target_percent']:>13.2f} | "
            f"{str(riga['in_tolleranza']):>6s} | "
            f"{riga['giorno_picco']:>11d} | "
            f"{riga['quota_cmax']:>18.3f}"
        )

    migliori_in_tolleranza = [r for r in risultati if r["in_tolleranza"]]
    if migliori_in_tolleranza:
        best = migliori_in_tolleranza[0]
        print("\nMiglior scenario IN tolleranza:")
    else:
        best = risultati[0]
        print("\nNessuno scenario entro tolleranza: migliore per distanza dal target:")

    print(
        f"  lambda={best['lambda']:.1f}, c_max={best['c_max']:.1f}, "
        f"picco={best['picco']:.0f} ({best['picco_percent']:.2f}%), "
        f"dist_target={best['distanza_target_percent']:.2f}%, "
        f"quota_cmax={best['quota_cmax']:.3f}"
    )

    return risultati


def esegui_scansione_target_picco_due_stadi(
    lambda_values_stage1,
    c_max_values_stage1,
    T_scan,
    target_picco_percent=10.0,
    tolleranza_percent=1.0,
    top_k_stage1=3,
    fattori_raffinamento_lambda=None,
    fattori_raffinamento_cmax=None,
):
    """
    Scansione target picco in due stadi.

    STADIO 1: griglia larga su (lambda, c_max).
    STADIO 2: raffinamento locale attorno ai top_k scenari dello stadio 1.

    L'ordinamento usa: distanza dal target -> quota tempo a c_max -> costo totale.
    """
    target_assoluto = (target_picco_percent / 100.0) * N
    key_sort = lambda x: (
        x["distanza_target_percent"],
        x["quota_cmax"],
        x["costo_totale"],
    )

    if fattori_raffinamento_lambda is None:
        fattori_raffinamento_lambda = [0.7, 0.85, 1.0, 1.15, 1.35]
    if fattori_raffinamento_cmax is None:
        fattori_raffinamento_cmax = [0.75, 0.9, 1.0, 1.1, 1.25]

    risultati_stage1 = []
    for lambda_val in lambda_values_stage1:
        for c_max_val in c_max_values_stage1:
            risultati_stage1.append(
                valuta_scenario_target_picco(
                    lambda_val,
                    c_max_val,
                    T_scan,
                    target_picco_percent,
                    tolleranza_percent,
                )
            )

    risultati_stage1.sort(key=key_sort)
    top_k_eff = max(1, min(top_k_stage1, len(risultati_stage1)))
    semi_stage2 = risultati_stage1[:top_k_eff]

    coppie_stage2 = set()
    for seed in semi_stage2:
        for f_lambda in fattori_raffinamento_lambda:
            for f_cmax in fattori_raffinamento_cmax:
                lambda_new = max(1e-8, float(seed["lambda"] * f_lambda))
                c_max_new = max(c_min_default, float(seed["c_max"] * f_cmax))
                coppie_stage2.add((round(lambda_new, 10), round(c_max_new, 10)))

    for seed in semi_stage2:
        coppie_stage2.add((round(float(seed["lambda"]), 10), round(float(seed["c_max"]), 10)))

    risultati_stage2 = []
    for lambda_val, c_max_val in sorted(coppie_stage2):
        risultati_stage2.append(
            valuta_scenario_target_picco(
                lambda_val,
                c_max_val,
                T_scan,
                target_picco_percent,
                tolleranza_percent,
            )
        )

    risultati_stage2.sort(key=key_sort)

    print("\n" + "=" * 116)
    print("SCAN CALIBRAZIONE TARGET PICCO A DUE STADI (lambda_reg_controllo, c_max)")
    print("=" * 116)
    print(
        f"Target picco: {target_picco_percent:.1f}% della popolazione "
        f"(~{target_assoluto:.0f} infetti), tolleranza +/-{tolleranza_percent:.1f}%"
    )
    print(f"Orizzonte scansione: T_scan={T_scan} giorni")
    print(
        f"Stadio 1: {len(lambda_values_stage1)}x{len(c_max_values_stage1)}={len(risultati_stage1)} scenari | "
        f"top_k={top_k_eff}"
    )
    print(f"Stadio 2: raffinamento locale su {len(coppie_stage2)} scenari")

    print("\nTop scenari STADIO 1 (primi 10):")
    print("lambda | c_max | picco_% | dist_target_% | in_tol | quota_tempo_a_cmax")
    print("-" * 84)
    for riga in risultati_stage1[:10]:
        print(
            f"{riga['lambda']:>8.3f} | "
            f"{riga['c_max']:>7.3f} | "
            f"{riga['picco_percent']:>7.2f} | "
            f"{riga['distanza_target_percent']:>13.2f} | "
            f"{str(riga['in_tolleranza']):>6s} | "
            f"{riga['quota_cmax']:>18.3f}"
        )

    print("\nTop scenari STADIO 2 (primi 12):")
    print("lambda | c_max | picco_% | dist_target_% | in_tol | quota_tempo_a_cmax")
    print("-" * 84)
    for riga in risultati_stage2[:12]:
        print(
            f"{riga['lambda']:>8.3f} | "
            f"{riga['c_max']:>7.3f} | "
            f"{riga['picco_percent']:>7.2f} | "
            f"{riga['distanza_target_percent']:>13.2f} | "
            f"{str(riga['in_tolleranza']):>6s} | "
            f"{riga['quota_cmax']:>18.3f}"
        )

    migliori_tol_stage2 = [r for r in risultati_stage2 if r["in_tolleranza"]]
    migliore_stage2 = migliori_tol_stage2[0] if migliori_tol_stage2 else risultati_stage2[0]

    print("\nMigliore scenario finale (STADIO 2):")
    print(
        f"  lambda={migliore_stage2['lambda']:.3f}, c_max={migliore_stage2['c_max']:.3f}, "
        f"picco={migliore_stage2['picco']:.0f} ({migliore_stage2['picco_percent']:.2f}%), "
        f"dist_target={migliore_stage2['distanza_target_percent']:.2f}%, "
        f"quota_cmax={migliore_stage2['quota_cmax']:.3f}, in_tolleranza={migliore_stage2['in_tolleranza']}"
    )

    return {
        "stage1": risultati_stage1,
        "stage2": risultati_stage2,
        "migliore_finale": migliore_stage2,
    }


# =============================================================================
# 7. Script di esecuzione (entrypoint)
# =============================================================================
# Blocco eseguibile: imposta i parametri di run, lancia le simulazioni di
# confronto e salva gli output grafici.

print("\n" + "=" * 80)
print("SIMULAZIONE STACKELBERG (Governo vs Cittadini)")
print("=" * 80)

a_logaritmica_calibrata = a_logaritmica_default
lambda_rischio_logaritmica_calibrata = lambda_rischio_logaritmica_default
rho_rischio_logaritmica_calibrato = rho_rischio_logaritmica_default

print("\n" + "=" * 80)
print("CONFRONTO FINALE IN SINGOLA RUN: QUADRATICA VS LOGARITMICA")
print("=" * 80)

S_quad, I_quad, R_quad, c_quad, x_bar_quad, x_star_quad, p_quad, log_quad = simula_sir_stackelberg_con_controllo_periodico(
    S_t0, I_t0, R_t0, N, beta, gamma, T,
    costo_infetto_giornaliero, k_saturazione_ospedali, alpha_saturazione_ospedali,
    c_iniziale=c_iniziale_default, intervallo_controllo=intervallo_controllo_default,
    orizzonte_predizione=orizzonte_predizione_default,
    c_min=c_min_default, c_max=c_max_default,
    soglia_attivazione_controllo=soglia_attivazione_controllo_default, fattore_isteresi=fattore_isteresi_default,
    lambda_reg_controllo=lambda_reg_controllo,
    kappa_prescrizione=kappa_prescrizione, rho_rischio=rho_rischio,
    eta_compliance=eta_compliance,
    tipo_best_response="quadratica",
    verbose_progress=verbose_progress_default,
)

S_log, I_log, R_log, c_log, x_bar_log, x_star_log, p_log, log_log = simula_sir_stackelberg_con_controllo_periodico(
    S_t0, I_t0, R_t0, N, beta, gamma, T,
    costo_infetto_giornaliero, k_saturazione_ospedali, alpha_saturazione_ospedali,
    c_iniziale=c_iniziale_default,
    intervallo_controllo=intervallo_controllo_default, orizzonte_predizione=orizzonte_predizione_default,
    c_min=c_min_default, c_max=c_max_default,
    soglia_attivazione_controllo=soglia_attivazione_controllo_default, fattore_isteresi=fattore_isteresi_default,
    lambda_reg_controllo=lambda_reg_controllo, kappa_prescrizione=kappa_prescrizione,
    rho_rischio=rho_rischio_logaritmica_default,
    eta_compliance=eta_compliance,
    a_logaritmica=a_logaritmica_default, epsilon_logaritmica=epsilon_logaritmica_default,
    lambda_rischio_logaritmica=lambda_rischio_logaritmica_default,
    num_grid_logaritmica=num_grid_logaritmica_default,
    tipo_best_response="logaritmica",
    verbose_progress=False,
)

t_picco_quad = int(I_quad.argmax())
t_picco_log = int(I_log.argmax())
picco_quad = float(I_quad.max())
picco_log = float(I_log.max())

costo_tot_quad = calcola_costo_epidemiologico_cumulato_con_controllo_variabile(
    S_quad, I_quad, costo_infetto_giornaliero, c_quad, k_saturazione_ospedali,
    alpha_saturazione_ospedali, 0, T, lambda_reg_controllo,
)
costo_tot_log = calcola_costo_epidemiologico_cumulato_con_controllo_variabile(
    S_log, I_log, costo_infetto_giornaliero, c_log, k_saturazione_ospedali,
    alpha_saturazione_ospedali, 0, T, lambda_reg_controllo,
)

compliance_media_quad = np.mean([x_star_quad[i] / max(x_bar_quad[i], 0.01) for i in range(len(x_star_quad))]) if T > 0 else np.nan
compliance_media_log = np.mean([x_star_log[i] / max(x_bar_log[i], 0.01) for i in range(len(x_star_log))]) if T > 0 else np.nan
gap_medio_quad = float(np.mean(x_star_quad - x_bar_quad))
gap_medio_log = float(np.mean(x_star_log - x_bar_log))
H_quad = ricostruisci_comparto_ospedalizzati_da_suscettibili(
    S_quad,
    h_ospedalizzazione=h_ospedalizzazione_default,
    tau_ospedalizzazione=tau_ospedalizzazione_default,
    gamma_ospedaliera=gamma_ospedaliera_default,
    H_t0=H_t0_default,
)
H_log = ricostruisci_comparto_ospedalizzati_da_suscettibili(
    S_log,
    h_ospedalizzazione=h_ospedalizzazione_default,
    tau_ospedalizzazione=tau_ospedalizzazione_default,
    gamma_ospedaliera=gamma_ospedaliera_default,
    H_t0=H_t0_default,
)
t_picco_H_quad = int(np.argmax(H_quad))
t_picco_H_log = int(np.argmax(H_log))
picco_H_quad = float(np.max(H_quad))
picco_H_log = float(np.max(H_log))

print("\nParametri utility logaritmica usati (default correnti):")
print(
    f"  a_logaritmica={a_logaritmica_calibrata:.3f}, "
    f"lambda_rischio_logaritmica={lambda_rischio_logaritmica_calibrata:.3f}, "
    f"rho_rischio={rho_rischio_logaritmica_calibrato:.3f}"
)

print("\nRisultati - Utility quadratica:")
print(f"  Picco infetti: {picco_quad:.0f} (giorno {t_picco_quad})")
print(f"  Picco ospedalizzati H: {picco_H_quad:.0f} (giorno {t_picco_H_quad})")
print(f"  Costo totale epidemico: {costo_tot_quad:,.2f}")
print(f"  Complianza media (x*/x_bar): {compliance_media_quad:.3f}")
print(f"  Gap medio (x* - x_bar): {gap_medio_quad:.4f}")

print("\nRisultati - Utility logaritmica:")
print(f"  Picco infetti: {picco_log:.0f} (giorno {t_picco_log})")
print(f"  Picco ospedalizzati H: {picco_H_log:.0f} (giorno {t_picco_H_log})")
print(f"  Costo totale epidemico: {costo_tot_log:,.2f}")
print(f"  Complianza media (x*/x_bar): {compliance_media_log:.3f}")
print(f"  Gap medio (x* - x_bar): {gap_medio_log:.4f}")

print("\nDelta logaritmica - quadratica:")
print(f"  Delta picco infetti: {picco_log - picco_quad:+.0f}")
print(f"  Delta costo totale: {costo_tot_log - costo_tot_quad:+,.2f}")
print(f"  Delta complianza media: {compliance_media_log - compliance_media_quad:+.3f}")

if verbose_progress_default:
    print(f"\nNumero ottimizzazioni MPC quadratica: {len(log_quad)}")
    print(f"Numero ottimizzazioni MPC logaritmica: {len(log_log)}")

plot_dinamica_compartimenti_stackelberg(
    t, S_quad, I_quad, R_quad, N, beta, gamma, R0, t_picco_quad,
    H=H_quad,
    simulation_label="quadratica",
    output_path="sir_compartimenti_stackelberg_quadratica.png",
)
plot_controllo_e_comportamento_stackelberg(
    t, S_quad, I_quad, c_quad, x_bar_quad, x_star_quad, p_quad,
    H=H_quad,
    simulation_label="quadratica",
    output_path="sir_stackelberg_analisi_quadratica.png",
)

plot_dinamica_compartimenti_stackelberg(
    t, S_log, I_log, R_log, N, beta, gamma, R0, t_picco_log,
    H=H_log,
    simulation_label="logaritmica",
    output_path="sir_compartimenti_stackelberg_logaritmica.png",
)
plot_controllo_e_comportamento_stackelberg(
    t, S_log, I_log, c_log, x_bar_log, x_star_log, p_log,
    H=H_log,
    simulation_label="logaritmica",
    output_path="sir_stackelberg_analisi_logaritmica.png",
)

if mostra_grafici_default:
    plt.show()
else:
    plt.close('all')


