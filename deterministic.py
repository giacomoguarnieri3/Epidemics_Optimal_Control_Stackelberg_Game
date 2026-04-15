"""
Modello Epidemiologico SIR — Deterministico con Processo di Markov
===================================================================

Il modello SIR suddivide la popolazione in tre compartimenti:
  - S (Suscettibili)
  - I (Infetti)     
  - R (Rimossi)     

La versione discreta a tempo discreto aggiorna ogni compartimento usando
solo il valore al passo precedente — questa è la proprietà di Markov:

    S_{t+1} = S_t  -  sigma(c,m) * beta * S_t * I_t / N
    I_{t+1} = I_t  +  sigma(c,m) * beta * S_t * I_t / N  -  gamma * I_t
    R_{t+1} = R_t  +  gamma * I_t

dove beta è il tasso di trasmissione e gamma il tasso di guarigione.
"""

# =============================================================================
# 1. Import delle librerie
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

print("Librerie importate con successo.")

# =============================================================================
# 2. Parametri del modello
# =============================================================================

# Parametri demografici ed epidemiologici
N        = 10_000     # Popolazione totale
I0       = 10         # Infetti iniziali
R0_init  = 0          # Guariti iniziali
S0       = N - I0 - R0_init  # Suscettibili iniziali

beta     = 0.3        # Tasso di trasmissione  (contatti efficaci per giorno)(covid: 0.3-0.5)(influenza: 0.1-0.3)
gamma    = 0.05       # Tasso di guarigione    (1/gamma = durata media malattia)

# Coefficienti del costo epidemiologico giornaliero
costo_infetto_giornaliero = 1000.0 #c_i
costo_controllo_per_suscettibile = 10.0 #c_s

# Parametri di saturazione ospedaliera nel costo degli infetti
k_saturazione_ospedali = 1000.0  
alpha_saturazione_ospedali = 0.2

# Parametro di efficacia della policy di controllo
m_controllo = 2.0

# Peso della penalizzazione quadratica del controllo nell'objective MPC
lambda_reg_controllo = 10.0

# Parametri simulazione controllo periodico (MPC)
intervallo_controllo_default = 80
orizzonte_predizione_default = 80
c_min_default = 0.05
c_max_default = 10000.0
soglia_attivazione_controllo_default = 0.005
fattore_isteresi_default = 0.0
verbose_progress_default = True

# Parametri numerici/operativi
num_grid_points_default = 401
step_percent_progress_default = 5

T        = 400        # Durata della simulazione (giorni)
t        = np.arange(0, T + 1)  # Asse temporale


# Numero riproduttivo di base
R0 = beta / gamma
print(f"Numero riproduttivo di base R0 = beta/gamma = {R0:.2f}")
print(f"Condizioni iniziali: S0={S0}, I0={I0}, R0_init={R0_init}, N={N}")

# =============================================================================
# 3. Simulazione deterministica a tempo discreto (Catena di Markov)
# =============================================================================
#
# Ogni passo temporale applica le equazioni alle differenze:
#
#   S_{t+1} = S_t  -  sigma(c,m) * beta * S_t * I_t / N          (nuovi infetti)
#   I_{t+1} = I_t  +  sigma(c,m) * beta * S_t * I_t / N  -  gamma * I_t   (guariti)
#   R_{t+1} = R_t  +  gamma * I_t
#
# La conservazione è garantita: S_t + I_t + R_t = N  per ogni t.


def sigma(c, m):
    """
    Efficacia della policy di controllo.

    Valori più alti di c riducono maggiormente il termine di infezione.
    """
    return 1 / (1 + m * c)


def stampa_progresso_simulazione(
    giorno_corrente,
    giorni_totali,
    label="Simulazione",
    step_percent=step_percent_progress_default,
):
    """
    Stampa lo stato di avanzamento della simulazione a step percentuali fissi.
    """
    if giorni_totali <= 0:
        return

    percentuale = int(np.floor(100 * giorno_corrente / giorni_totali))
    if percentuale >= 100 or percentuale % step_percent == 0:
        print(f"[{label}] progresso: {percentuale:>3}% (giorno {giorno_corrente}/{giorni_totali})")


def sir_markov_deterministic(
    S0,
    I0,
    R0_init,
    N,
    beta,
    gamma,
    T,
    costo_controllo_per_suscettibile,
    m_controllo,
):
    """
    Simula il modello SIR deterministico a tempo discreto.

    La transizione di stato è un processo di Markov:
    lo stato al tempo t+1 dipende SOLO dallo stato al tempo t.

    Parametri
    ----------
    S0, I0, R0_init : condizioni iniziali dei compartimenti
    N               : popolazione totale
    beta            : tasso di trasmissione
    gamma           : tasso di guarigione
    T               : numero di passi temporali
    costo_controllo_per_suscettibile : spesa pubblica per suscettibile (parametro c della policy)
    m_controllo     : parametro di efficacia della policy

    Ritorna
    -------
    S, I, R : array numpy di lunghezza T+1
    """
    S = np.zeros(T + 1)
    I = np.zeros(T + 1)
    R = np.zeros(T + 1)

    # Condizioni iniziali
    S[0], I[0], R[0] = S0, I0, R0_init

    for step in range(T):
        # ---- Flussi di transizione (solo dai valori al passo t) ----
        efficacia_controllo = sigma(costo_controllo_per_suscettibile, m_controllo)
        new_infections = efficacia_controllo * beta * S[step] * I[step] / N   # S → I
        new_recoveries = gamma * I[step]                 # I → R

        # ---- Aggiornamento compartimenti (Markov step) ----
        S[step + 1] = S[step] - new_infections
        I[step + 1] = I[step] + new_infections - new_recoveries
        R[step + 1] = R[step] + new_recoveries

    return S, I, R


def calcola_costo_epidemiologico_istantaneo(
    S_t,
    I_t,
    costo_infetto_giornaliero,
    costo_controllo_per_suscettibile,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
):
    """
    Calcola il costo epidemiologico giornaliero associato allo stato corrente.

    Il costo istantaneo è definito come:

        costo(t) = costo_infetto_giornaliero * I_t
                 + k_saturazione_ospedali * exp(alpha_saturazione_ospedali * I_t)
                 + costo_controllo_per_suscettibile * S_t
    """
    return (
        costo_infetto_giornaliero * I_t
        + k_saturazione_ospedali * np.exp(alpha_saturazione_ospedali * I_t)
        + costo_controllo_per_suscettibile * S_t
    )


def calcola_costo_epidemiologico_cumulato(
    S,
    I,
    costo_infetto_giornaliero,
    costo_controllo_per_suscettibile,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    t0,
    giorno_corrente,
):
    """
    Calcola il costo epidemiologico cumulato tra i giorni t0 e giorno_corrente.

    Il costo giornaliero allo stato t è definito come:

        costo(t) = costo_infetto_giornaliero * I[t]
                 + k_saturazione_ospedali * exp(alpha_saturazione_ospedali * I[t])
                 + costo_controllo_per_suscettibile * S[t]

    Il costo cumulato è quindi:

        somma_{t=t0}^{giorno_corrente} costo(t)
    """
    if t0 < 0:
        raise ValueError("t0 deve essere maggiore o uguale a 0.")

    if giorno_corrente < t0:
        raise ValueError("giorno_corrente deve essere maggiore o uguale a t0.")

    if giorno_corrente >= len(S) or giorno_corrente >= len(I):
        raise ValueError("giorno_corrente eccede la lunghezza delle traiettorie S o I.")

    costi_giornalieri = [
        calcola_costo_epidemiologico_istantaneo(
            S[giorno],
            I[giorno],
            costo_infetto_giornaliero,
            costo_controllo_per_suscettibile,
            k_saturazione_ospedali,
            alpha_saturazione_ospedali,
        )
        for giorno in range(t0, giorno_corrente + 1)
    ]

    return np.sum(costi_giornalieri)


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
):
    """
    Calcola il costo cumulato quando c_s varia nel tempo.

    c_schedule è il valore di controllo applicato nel passaggio giorno -> giorno+1.
    Per il costo allo stato del giorno t si usa c_schedule[t].

    Il termine di controllo nel costo è modellato come penalizzazione
    quadratica: lambda_reg_controllo * c_schedule[t]^2.
    """
    if t0 < 0:
        raise ValueError("t0 deve essere maggiore o uguale a 0.")

    if giorno_corrente < t0:
        raise ValueError("giorno_corrente deve essere maggiore o uguale a t0.")

    if giorno_corrente >= len(S) or giorno_corrente >= len(I):
        raise ValueError("giorno_corrente eccede la lunghezza delle traiettorie S o I.")

    if len(c_schedule) == 0:
        raise ValueError("c_schedule non può essere vuoto.")

    costi_giornalieri = [
        (
            calcola_costo_epidemiologico_istantaneo(
                S[giorno],
                I[giorno],
                costo_infetto_giornaliero,
                0.0,
                k_saturazione_ospedali,
                alpha_saturazione_ospedali,
            )
            + lambda_reg_controllo * c_schedule[min(giorno, len(c_schedule) - 1)] ** 2
        )
        for giorno in range(t0, giorno_corrente + 1)
    ]

    return np.sum(costi_giornalieri)


def plot_dinamica_compartimenti(t, S, I, R, N, beta, gamma, R0, t_picco):
    """
    Visualizza la dinamica dei compartimenti S, I, R e l'incidenza giornaliera.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Modello SIR con Controllo MPC — Catena di Markov a Tempo Discreto\n"
        f"$N={N}$,  $\\beta={beta}$,  $\\gamma={gamma}$,  $R_0={R0:.1f}$",
        fontsize=14,
        fontweight="bold",
    )

    # --- Grafico superiore: tutti e tre i compartimenti ---
    ax1 = axes[0]
    ax1.plot(t, S / N * 100, color="steelblue", lw=2, label="S — Suscettibili")
    ax1.plot(t, I / N * 100, color="crimson", lw=2, label="I — Infetti (con MPC)")
    ax1.plot(t, R / N * 100, color="forestgreen", lw=2, label="R — Rimossi/Guariti")

    ax1.axvline(t_picco, color="crimson", lw=1.2, ls="--", alpha=0.6,
                label=f"Picco I (giorno {t_picco})")
    ax1.set_ylabel("Frazione di popolazione (%)", fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Evoluzione dei compartimenti nel tempo", fontsize=11)

    # --- Grafico inferiore: nuovi infetti per giorno ---
    ax2 = axes[1]
    nuovi_infetti = beta * S[:-1] * I[:-1] / N   # flusso S→I al passo t
    ax2.bar(t[:-1], nuovi_infetti, color="crimson", alpha=0.6, width=1,
            label="Nuovi infetti / giorno")
    ax2.set_xlabel("Tempo (giorni)", fontsize=11)
    ax2.set_ylabel("Nuovi infetti", fontsize=11)
    ax2.set_title("Incidenza giornaliera (flusso S → I)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sir_compartimenti.png", dpi=150)


def simula_finestra_predizione(
    S_init,
    I_init,
    R_init,
    N,
    beta,
    gamma,
    orizzonte,
    c_s,
    m_controllo,
):
    """
    Simula una predizione SIR su una finestra temporale di lunghezza orizzonte,
    mantenendo c_s costante durante tutta la finestra.
    """
    S_pred = np.zeros(orizzonte + 1)
    I_pred = np.zeros(orizzonte + 1)
    R_pred = np.zeros(orizzonte + 1)
    S_pred[0], I_pred[0], R_pred[0] = S_init, I_init, R_init

    efficacia_controllo = sigma(c_s, m_controllo)
    for k in range(orizzonte):
        new_infections = efficacia_controllo * beta * S_pred[k] * I_pred[k] / N
        new_recoveries = gamma * I_pred[k]

        S_pred[k + 1] = S_pred[k] - new_infections
        I_pred[k + 1] = I_pred[k] + new_infections - new_recoveries
        R_pred[k + 1] = R_pred[k] + new_recoveries

    return S_pred, I_pred, R_pred


def costo_previsto_su_finestra(
    S_init,
    I_init,
    R_init,
    N,
    beta,
    gamma,
    orizzonte,
    costo_infetto_giornaliero,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    c_s,
    m_controllo,
    lambda_reg_controllo,
):
    """
    Calcola il costo cumulato previsto su una finestra usando c_s costante.

    L'objective usa penalizzazione quadratica del controllo:
    J = costo_epidemico + lambda_reg_controllo * c_s^2 * (orizzonte + 1)
    """
    S_pred, I_pred, _ = simula_finestra_predizione(
        S_init,
        I_init,
        R_init,
        N,
        beta,
        gamma,
        orizzonte,
        c_s,
        m_controllo,
    )

    costo_epidemico = calcola_costo_epidemiologico_cumulato(
        S_pred,
        I_pred,
        costo_infetto_giornaliero,
        0.0,
        k_saturazione_ospedali,
        alpha_saturazione_ospedali,
        t0=0,
        giorno_corrente=orizzonte,
    )

    costo_controllo = lambda_reg_controllo * c_s**2 * (orizzonte + 1)
    return costo_epidemico + costo_controllo


def ottimizza_c_s_su_finestra(
    S_init,
    I_init,
    R_init,
    N,
    beta,
    gamma,
    orizzonte,
    costo_infetto_giornaliero,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    c_iniziale,
    m_controllo,
    c_min,
    c_max,
    lambda_reg_controllo,
):
    """
    Ottimizza c_s su una singola finestra temporale con ricerca su griglia.

    Nota: essendo c_s scalare, la ricerca diretta su [c_min, c_max] e' piu robusta
    del gradiente proiettato in presenza di funzioni costo molto ripide o mal
    condizionate (evita blocchi artificiali su c_min/c_max).
    """
    c_iniziale_clip = float(np.clip(c_iniziale, c_min, c_max))
    J_iniziale = costo_previsto_su_finestra(
        S_init,
        I_init,
        R_init,
        N,
        beta,
        gamma,
        orizzonte,
        costo_infetto_giornaliero,
        k_saturazione_ospedali,
        alpha_saturazione_ospedali,
        c_iniziale_clip,
        m_controllo,
        lambda_reg_controllo,
    )

    num_grid_points = num_grid_points_default
    griglia_c = np.linspace(c_min, c_max, num_grid_points)
    valori_J = np.array([
        costo_previsto_su_finestra(
            S_init,
            I_init,
            R_init,
            N,
            beta,
            gamma,
            orizzonte,
            costo_infetto_giornaliero,
            k_saturazione_ospedali,
            alpha_saturazione_ospedali,
            c_val,
            m_controllo,
            lambda_reg_controllo,
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

    last_delta_costo_assoluto = abs(J_corrente - J_iniziale)
    last_delta_costo_relativo = last_delta_costo_assoluto / (abs(J_iniziale) + 1e-12)

    info = {
        "iterations": iter_done,
        "stop_reason": stop_reason,
        "last_delta_costo_assoluto": float(last_delta_costo_assoluto),
        "last_delta_costo_relativo": float(last_delta_costo_relativo),
        "c_ottimo": float(c_corrente),
        "J_ottimo": float(J_corrente),
    }
    return c_corrente, info


def simula_sir_con_controllo_periodico(
    S0,
    I0,
    R0_init,
    N,
    beta,
    gamma,
    T,
    costo_infetto_giornaliero,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    c_iniziale,
    m_controllo,
    intervallo_controllo=intervallo_controllo_default,
    orizzonte_predizione=orizzonte_predizione_default,
    c_min=c_min_default,
    c_max=c_max_default,
    soglia_attivazione_controllo=soglia_attivazione_controllo_default,
    fattore_isteresi=fattore_isteresi_default,
    lambda_reg_controllo=lambda_reg_controllo,
    verbose_progress=verbose_progress_default,
):
    """
    Simula il sistema reale con attivazione del controllo su soglia infetti.

     Regola operativa:
    1) finché I(t)/N < soglia_attivazione_controllo, non ottimizza e applica c_s=c_min;
     2) alla prima violazione della soglia, esegue la prima ottimizzazione;
     3) poi aggiorna c_s ogni intervallo_controllo giorni;
    4) quando I(t)/N torna sotto soglia, disattiva l'MPC e applica c_s=c_min.
    """
    S = np.zeros(T + 1)
    I = np.zeros(T + 1)
    R = np.zeros(T + 1)
    S[0], I[0], R[0] = S0, I0, R0_init

    c_schedule = np.zeros(T)
    log_ottimizzazione = []

    giorno = 0
    c_guess = float(np.clip(c_iniziale, c_min, c_max))
    ultimo_checkpoint_stampato = -1
    controllo_attivo = False
    prossimo_giorno_ottimizzazione = None
    c_applicato = c_min
    sigma_run = sigma(c_min, m_controllo)
    soglia_infetti_assoluta = soglia_attivazione_controllo * N
    soglia_bassa = soglia_infetti_assoluta * fattore_isteresi

    if verbose_progress:
        print("\n[Controllo periodico] avvio simulazione...")
        stampa_progresso_simulazione(0, T, label="Controllo periodico")

    while giorno < T:
        if (not controllo_attivo) and (I[giorno] >= soglia_infetti_assoluta):
            controllo_attivo = True
            prossimo_giorno_ottimizzazione = giorno

        if controllo_attivo and (I[giorno] < soglia_bassa):
            controllo_attivo = False
            prossimo_giorno_ottimizzazione = None
            c_applicato = c_min
            sigma_run = sigma(c_min, m_controllo)

        if controllo_attivo and (prossimo_giorno_ottimizzazione is not None) and (giorno >= prossimo_giorno_ottimizzazione):
            orizzonte_ottimizzazione = min(orizzonte_predizione, T - giorno)

            c_ottimo, info = ottimizza_c_s_su_finestra(
                S[giorno],
                I[giorno],
                R[giorno],
                N,
                beta,
                gamma,
                orizzonte_ottimizzazione,
                costo_infetto_giornaliero,
                k_saturazione_ospedali,
                alpha_saturazione_ospedali,
                c_guess,
                m_controllo,
                c_min,
                c_max,
                lambda_reg_controllo,
            )

            c_applicato = c_ottimo
            sigma_run = sigma(c_ottimo, m_controllo)
            info["giorno_start"] = int(giorno)
            info["giorno_end"] = int(min(giorno + intervallo_controllo, T))
            info["orizzonte_predizione"] = int(orizzonte_ottimizzazione)
            info["sigma_ottima"] = float(sigma_run)
            log_ottimizzazione.append(info)

            c_guess = c_ottimo
            prossimo_giorno_ottimizzazione = giorno + intervallo_controllo
        elif not controllo_attivo:
            c_applicato = c_min
            sigma_run = sigma(c_min, m_controllo)

        new_infections = sigma_run * beta * S[giorno] * I[giorno] / N
        new_recoveries = gamma * I[giorno]

        S[giorno + 1] = S[giorno] - new_infections
        I[giorno + 1] = I[giorno] + new_infections - new_recoveries
        R[giorno + 1] = R[giorno] + new_recoveries
        c_schedule[giorno] = c_applicato

        if verbose_progress:
            percentuale = int(np.floor(100 * (giorno + 1) / T))
            checkpoint = percentuale // 5
            if checkpoint > ultimo_checkpoint_stampato:
                ultimo_checkpoint_stampato = checkpoint
                stampa_progresso_simulazione(giorno + 1, T, label="Controllo periodico")

        giorno += 1

    if verbose_progress:
        print("[Controllo periodico] simulazione completata.")

    return S, I, R, c_schedule, log_ottimizzazione


def plot_controllo_periodico(t, S, I, c_schedule, costo_infetto_giornaliero):
    """
    Plotta sull'intera simulazione i profili economici associati al controllo.

    Vengono mostrati:
    1) il controllo c_s(t),
    2) il coefficiente costante c_i,
    3) il prodotto c_s(t) * S(t),
    4) il prodotto c_i * I(t).
    """
    t_controllo = t[:-1]
    S_eff = S[:-1]
    I_eff = I[:-1]
    costo_controllo_istantaneo = c_schedule * S_eff
    costo_infetti_istantaneo = costo_infetto_giornaliero * I_eff

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    ax1.step(t_controllo, c_schedule, where="post", lw=2, color="purple", label="c_s(t)")
    ax1.set_ylabel("c_s")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t_controllo, I_eff / I_eff[0:1].sum() * 0 + I_eff, lw=2, color="crimson", label="I(t) con controllo MPC")
    ax2.plot(t_controllo, np.full_like(t_controllo, soglia_attivazione_controllo_default * N, dtype=float),
             lw=1, ls="--", color="orange", label=f"Soglia alta ({int(soglia_attivazione_controllo_default * N)})")
    ax2.plot(t_controllo, np.full_like(t_controllo, soglia_attivazione_controllo_default * N * fattore_isteresi_default, dtype=float),
             lw=1, ls=":", color="gold", label=f"Soglia bassa ({int(soglia_attivazione_controllo_default * N * fattore_isteresi_default)})")
    ax2.set_ylabel("I(t) [individui]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(t_controllo, costo_controllo_istantaneo, lw=2, color="steelblue", label="c_s(t) · S(t)")
    ax3.set_ylabel("c_s · S")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(t_controllo, costo_infetti_istantaneo, lw=2, color="crimson", label="c_i · I(t)")
    ax4.set_xlabel("Tempo (giorni)")
    ax4.set_ylabel("c_i · I")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.suptitle(
        "Profili temporali di controllo e costo sull'intera simulazione",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("sir_controllo_periodico.png", dpi=150)


def analizza_proprieta_markov(N, beta, gamma, R0, S_det, I_det, R_det, conservazione):
    """
    Stampa una sintesi delle principali proprietà del modello SIR markoviano.
    """
    print("=" * 60)
    print("  PROPRIETÀ DELLA CATENA DI MARKOV SIR")
    print("=" * 60)

    print(f"\n{'Parametro':<35} {'Valore':>10}")
    print("-" * 50)
    print(f"{'Popolazione totale N':<35} {N:>10,}")
    print(f"{'Tasso trasmissione β':<35} {beta:>10.3f}")
    print(f"{'Tasso guarigione γ':<35} {gamma:>10.3f}")
    print(f"{'Numero riproduttivo di base R0':<35} {R0:>10.2f}")
    print(f"{'Durata media malattia (1/γ)':<35} {1/gamma:>10.1f} giorni")

    print(f"\n{'--- Risultati della simulazione ---':^50}")
    print(f"{'Picco di infetti':<35} {I_det.max():>10.0f} individui")
    print(f"{'Giorno del picco':<35} {I_det.argmax():>10} ")
    pct_infettati = (1 - S_det[-1] / N) * 100
    print(f"{'Totale infettati (% pop.)':<35} {pct_infettati:>10.1f} %")
    print(f"{'Suscettibili rimasti (% pop.)':<35} {S_det[-1]/N*100:>10.1f} %")

    print(f"\n{'--- Proprietà di Markov ---':^50}")
    print("\n  Proprietà di Markov: lo stato (S,I,R) al tempo t+1")
    print("    dipende SOLO dallo stato al tempo t.")
    print("\n  Stazionarietà dell'epidemia: lo stato finale (S_inf, 0, R_inf)")
    print("    è un punto fisso — la catena converge a uno stato assorbente.")
    print("\n  Conservazione: S(t) + I(t) + R(t) = N per ogni t.")
    print(f"\n  Max errore numerico conservazione: {conservazione:.2e}")

# =============================================================================
# 4. Controllo periodico condizionato su soglia infetti
# =============================================================================
S_ctrl, I_ctrl, R_ctrl, c_schedule_ctrl, log_ctrl = simula_sir_con_controllo_periodico(
    S0,
    I0,
    R0_init,
    N,
    beta,
    gamma,
    T,
    costo_infetto_giornaliero,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    c_iniziale=costo_controllo_per_suscettibile,
    m_controllo=m_controllo,
    intervallo_controllo=intervallo_controllo_default,
    orizzonte_predizione=orizzonte_predizione_default,
    c_min=c_min_default,
    c_max=c_max_default,
    soglia_attivazione_controllo=soglia_attivazione_controllo_default,
    fattore_isteresi=fattore_isteresi_default,
    lambda_reg_controllo=lambda_reg_controllo,
    verbose_progress=verbose_progress_default,
)

print("\n" + "=" * 80)
print(
    "RISULTATI CONTROLLO PERIODICO "
    f"(finestra = {intervallo_controllo_default} giorni, "
    f"trigger su soglia {soglia_attivazione_controllo_default * 100:.1f}%)"
)
print("=" * 80)
for voce in log_ctrl:
    print(
        f"giorni [{voce['giorno_start']:>3},{voce['giorno_end']:>3}] | "
        f"Hpred= {voce['orizzonte_predizione']:>3} | "
        f"c*= {voce['c_ottimo']:.4f} | sigma*= {voce['sigma_ottima']:.4f} | "
        f"iter= {voce['iterations']:>2} | stop= {voce['stop_reason']:<9} | "
        f"dJ_abs= {voce['last_delta_costo_assoluto']:.6f} | "
        f"dJ_rel= {voce['last_delta_costo_relativo']:.6e}"
    )

if len(log_ctrl) > 0:
    ultima_diff_abs = log_ctrl[-1]["last_delta_costo_assoluto"]
    ultima_diff_rel = log_ctrl[-1]["last_delta_costo_relativo"]
    print(f"\nUltima differenza assoluta tra funzionali: {ultima_diff_abs:.6f}")
    print(f"Ultima differenza relativa tra funzionali: {ultima_diff_rel:.6e}")

costo_totale_ctrl = calcola_costo_epidemiologico_cumulato_con_controllo_variabile(
    S_ctrl,
    I_ctrl,
    costo_infetto_giornaliero,
    c_schedule_ctrl,
    k_saturazione_ospedali,
    alpha_saturazione_ospedali,
    t0=0,
    giorno_corrente=T,
    lambda_reg_controllo=lambda_reg_controllo,
)
print(f"Costo cumulato con controllo periodico: {costo_totale_ctrl:,.2f}")
t_picco_ctrl = int(I_ctrl.argmax())
conservazione_ctrl = np.max(np.abs(S_ctrl + I_ctrl + R_ctrl - N))
print(f"Verifica conservazione (S+I+R=N): errore massimo = {conservazione_ctrl:.2e}")
print(f"Picco infettivi (con MPC): {I_ctrl.max():.0f} individui al giorno {t_picco_ctrl}")

# =============================================================================
# 5. Analisi delle proprietà della catena di Markov (simulazione con MPC)
# =============================================================================

analizza_proprieta_markov(N, beta, gamma, R0, S_ctrl, I_ctrl, R_ctrl, conservazione_ctrl)

# =============================================================================
# 6. Visualizzazione: SIR controllato + profili economici
# =============================================================================

plot_dinamica_compartimenti(t, S_ctrl, I_ctrl, R_ctrl, N, beta, gamma, R0, t_picco_ctrl)

plot_controllo_periodico(
    t,
    S_ctrl,
    I_ctrl,
    c_schedule_ctrl,
    costo_infetto_giornaliero,
)

plt.show()
