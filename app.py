import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===========================
# PINN per carica RC (Vs, tau)
# ===========================

class PINN_RC_Charging(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Parametri fisici incogniti: Vs e tau (log_tau -> tau > 0)
        Vs_init = 4.0          # valore di partenza
        log_tau_init = np.log(0.05)  # valore di partenza per tau

        self.Vs_param = nn.Parameter(torch.tensor(Vs_init, dtype=torch.float32))
        self.log_tau = nn.Parameter(torch.tensor(log_tau_init, dtype=torch.float32))

    def forward(self, t):
        return self.net(t)

    def dVdt(self, t):
        t_req = t.clone().detach().requires_grad_(True)
        V = self.forward(t_req)
        dVdt = torch.autograd.grad(
            V, t_req,
            grad_outputs=torch.ones_like(V),
            retain_graph=True,
            create_graph=True
        )[0]
        return dVdt

    def Vs(self):
        return self.Vs_param

    def tau(self):
        return torch.exp(self.log_tau)


def loss_fn(model, t_data, V_data, w_data=1.0, w_pde=20.0, w_ic=0.1):
    # 1) Loss sui dati
    V_pred = model(t_data)
    loss_data = torch.mean((V_pred - V_data) ** 2)

    # 2) Loss fisica (PDE carica): dV/dt - (1/tau)*(Vs - V) = 0
    # Usiamo solo i tempi iniziali per la PDE (più informativi)
    t_phys = t_data[t_data[:, 0] < 0.25]  # primi 250 ms
    dVdt = model.dVdt(t_phys)
    V_pred_phys = model(t_phys)
    Vs_current = model.Vs()
    tau_current = model.tau()
    PDE_residual = dVdt - (1.0 / tau_current) * (Vs_current - V_pred_phys)
    loss_pde = torch.mean(PDE_residual ** 2)

    # 3) Condizione iniziale: media sui primi istanti
    idx_ic = t_data[:, 0] < 0.01  # primi 10 ms
    V0_pred = model(t_data[idx_ic])
    V0_target = V_data[idx_ic]
    loss_ic = torch.mean((V0_pred - V0_target) ** 2)

    loss_total = w_data * loss_data + w_pde * loss_pde + w_ic * loss_ic
    return loss_total, loss_data.detach(), loss_pde.detach(), loss_ic.detach()


# ===========================
# Funzioni di utilità
# ===========================

def generate_synthetic_data(Vs_true=5.0, tau_true=0.1, t_end=0.5, dt=0.001, noise_std=0.05):
    t = np.arange(0, t_end + dt, dt)
    V_clean = Vs_true * (1.0 - np.exp(-t / tau_true))
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, noise_std, size=t.shape)
    V_meas = V_clean + noise
    return t, V_clean, V_meas, Vs_true, tau_true


def load_csv(file):
    df = pd.read_csv(file)

    # Proviamo a indovinare i nomi delle colonne
    possible_t = ['t', 'time', 'tempo', 't_s']
    possible_v = ['V', 'v', 'voltage', 'tensione', 'V_meas_V']

    t_col = None
    v_col = None

    for c in df.columns:
        if c in possible_t:
            t_col = c
        if c in possible_v:
            v_col = c

    if t_col is None or v_col is None:
        st.error("Impossibile riconoscere le colonne tempo/tensione. "
                 "Assicurati che il CSV contenga ad esempio colonne chiamate 't' e 'V'.")
        return None, None

    t = df[t_col].values
    V_meas = df[v_col].values

    return t, V_meas


# ===========================
# Streamlit App
# ===========================

def main():
    st.title("PINN per circuito RC in carica")
    st.write("Stima della tensione del generatore **Vs** e della costante di tempo **τ** "
             "a partire da dati (tempo, tensione sul condensatore).")

    st.sidebar.header("Input dati")

    mode = st.sidebar.radio(
        "Scegli origine dei dati:",
        ("Dati sintetici (simulati)", "Carica CSV esterno")
    )

    if mode == "Dati sintetici (simulati)":
        Vs_true = st.sidebar.slider("Vs vero [V]", 1.0, 10.0, 5.0, 0.1)
        tau_true = st.sidebar.slider("tau vero [s]", 0.01, 0.5, 0.10, 0.01)
        noise_std = st.sidebar.slider("Rumore (σ) [V]", 0.0, 0.2, 0.05, 0.01)
        t_end = st.sidebar.slider("Tempo finale [s]", 0.1, 2.0, 0.5, 0.1)
        dt = st.sidebar.slider("Passo temporale [s]", 0.0005, 0.01, 0.001, 0.0005)

        t, V_clean, V_meas, Vs_true, tau_true = generate_synthetic_data(
            Vs_true=Vs_true,
            tau_true=tau_true,
            t_end=t_end,
            dt=dt,
            noise_std=noise_std
        )
        st.success(f"Dati sintetici generati: Vs_true = {Vs_true:.3f} V, tau_true = {tau_true:.4f} s")

    else:
        uploaded_file = st.sidebar.file_uploader("Carica CSV con colonne tempo e tensione", type=["csv"])
        if uploaded_file is None:
            st.warning("Carica un file CSV per continuare.")
            st.stop()
        t, V_meas = load_csv(uploaded_file)
        if t is None:
            st.stop()
        V_clean = None  # non conosciamo la curva ideale
        Vs_true = None
        tau_true = None
        st.success("Dati caricati dal CSV.")

    # Mostra dati grezzi
    st.subheader("Dati di ingresso")
    fig_data, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(t, V_meas, s=5, alpha=0.5, label="V misurata")
    if V_clean is not None:
        ax.plot(t, V_clean, label="V ideale (segnale vero)")
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel("Tensione sul condensatore [V]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_data)

    # Preparazione dati per PyTorch
    t_data = torch.tensor(t, dtype=torch.float32).view(-1, 1)
    V_data = torch.tensor(V_meas, dtype=torch.float32).view(-1, 1)

    st.sidebar.header("Parametri di training")
    n_epochs = st.sidebar.slider("Numero di epoche", 1000, 20000, 8000, 1000)
    lr_net = st.sidebar.select_slider("Learning rate rete", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
    lr_phys = st.sidebar.select_slider("Learning rate parametri fisici", options=[1e-4, 5e-4, 1e-3], value=5e-4)

    if st.button("Avvia training PINN"):
        model = PINN_RC_Charging()
        optimizer = torch.optim.Adam([
            {"params": model.net.parameters(), "lr": lr_net},
            {"params": [model.Vs_param, model.log_tau], "lr": lr_phys},
        ])

        history = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            loss, ld, lp, lic = loss_fn(model, t_data, V_data)
            loss.backward()
            optimizer.step()

            history.append([epoch, loss.item(), ld.item(), lp.item(), lic.item()])

            if epoch % 200 == 0 or epoch == 1:
                Vs_est = model.Vs().item()
                tau_est = model.tau().item()
                status_text.text(
                    f"Epoch {epoch}/{n_epochs} | Loss = {loss.item():.3e} | "
                    f"Vs = {Vs_est:.3f} V, tau = {tau_est:.4f} s"
                )
            progress_bar.progress(epoch / n_epochs)

        st.success("Training completato!")

        Vs_est = model.Vs().item()
        tau_est = model.tau().item()
        st.subheader("Parametri stimati dalla PINN")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Vs stimato** = {Vs_est:.3f} V")
            if Vs_true is not None:
                st.write(f"Vs vero (sintetico) = {Vs_true:.3f} V")
        with col2:
            st.write(f"**tau stimato** = {tau_est:.4f} s")
            if tau_true is not None:
                st.write(f"tau vero (sintetico) = {tau_true:.4f} s")

        # Curva PINN vs dati
        model.eval()
        with torch.no_grad():
            V_pred_final = model(t_data).numpy().flatten()

        st.subheader("Confronto tra dati, PINN e (se disponibile) curva ideale")
        fig_fit, ax2 = plt.subplots(figsize=(8, 4))
        ax2.scatter(t, V_meas, s=5, alpha=0.5, label="V misurata")
        if V_clean is not None:
            ax2.plot(t, V_clean, label="V ideale (vero)")
        ax2.plot(t, V_pred_final, linestyle="--", label="V PINN")
        ax2.set_xlabel("Tempo [s]")
        ax2.set_ylabel("Tensione [V]")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig_fit)

        # Andamento della loss
        history = np.array(history)
        epochs = history[:, 0]
        loss_tot = history[:, 1]

        st.subheader("Convergenza della loss (Adam)")
        fig_loss, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(epochs, loss_tot)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss totale")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig_loss)

    else:
        st.info("Imposta i parametri nella sidebar e premi **'Avvia training PINN'** per iniziare.")


if __name__ == "__main__":
    main()