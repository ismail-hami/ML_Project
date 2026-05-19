# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:55:04 2026

@author: Hp
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings('ignore')

# ─── EMSI LIGHT PALETTE ────────────────────────────────────────────────────────
BG_MAIN    = "#F5F7FA"       # very light grey-white page background
BG_CARD    = "#FFFFFF"       # pure white cards
BG_PANEL   = "#EEF2F7"      # sidebar background
EMSI_GREEN = "#00843D"       # EMSI primary green
EMSI_DGRN  = "#005C2A"       # darker green for hover / headers
EMSI_LGRN  = "#E6F4EC"       # light green tint
ACCENT1    = "#00843D"       # régression  – green
ACCENT2    = "#0072C6"       # clustering  – blue
ACCENT3    = "#E63946"       # RF          – red
ACCENT4    = "#8338EC"       # ARIMA       – purple
ACCENT5    = "#F77F00"       # NN          – orange
ACCENT6    = "#2EC4B6"       # CV          – teal
TEXT_MAIN  = "#1A1A2E"       # near-black text
TEXT_SUB   = "#5A6A7E"       # secondary grey text
BORDER     = "#D0DAE8"       # subtle border
BTN_HOVER  = "#E6F4EC"       # light-green hover

FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_SUB   = ("Segoe UI", 10)
FONT_BTN   = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_CODE  = ("Consolas", 9)

ALGO_COLORS = {
    "Régression Linéaire": ACCENT1,
    "Clustering":          ACCENT2,
    "Random Forest":       ACCENT3,
    "Time Series ARIMA":   ACCENT4,
    "Réseaux de Neurones": ACCENT5,
    "Validation Croisée":  ACCENT6,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
class ScrollFrame(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG_MAIN, **kw)
        self.canvas = tk.Canvas(self, bg=BG_MAIN, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=BG_MAIN)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", self._on_scroll)

    def _on_scroll(self, e):
        self.canvas.yview_scroll(int(-1*(e.delta/120)), "units")


def labeled_entry(parent, text, default, color=ACCENT1):
    f = tk.Frame(parent, bg=BG_PANEL)
    f.pack(fill="x", pady=3)
    tk.Label(f, text=text, bg=BG_PANEL, fg=TEXT_SUB,
             font=FONT_SMALL, width=20, anchor="w").pack(side="left")
    var = tk.StringVar(value=str(default))
    e = tk.Entry(f, textvariable=var, bg=BG_CARD, fg=TEXT_MAIN,
                 insertbackground=color, relief="flat",
                 font=FONT_SMALL, width=8,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=color)
    e.pack(side="left", padx=4)
    return var

def section_label(parent, text, color):
    f = tk.Frame(parent, bg=BG_PANEL)
    f.pack(fill="x", pady=(10, 2))
    tk.Frame(f, bg=color, width=3).pack(side="left", fill="y", padx=(0, 6))
    tk.Label(f, text=text, bg=BG_PANEL, fg=color,
             font=("Segoe UI", 9, "bold")).pack(side="left", anchor="w")

def run_button(parent, text, cmd, color):
    f = tk.Frame(parent, bg=color, pady=1)
    f.pack(pady=12, fill="x", padx=10)
    b = tk.Button(f, text=text, command=cmd,
                  bg=color, fg="white", font=FONT_BTN,
                  relief="flat", cursor="hand2",
                  activebackground=EMSI_DGRN, activeforeground="white",
                  padx=14, pady=8)
    b.pack(fill="x")
    b.bind("<Enter>", lambda e: b.config(bg=EMSI_DGRN if color==EMSI_GREEN else color))
    b.bind("<Leave>", lambda e: b.config(bg=color))
    return b

def metrics_box(parent, lines, color):
    f = tk.Frame(parent, bg=BG_CARD,
                 highlightthickness=1, highlightbackground=color)
    f.pack(fill="x", pady=6, padx=6)
    # colored top bar
    tk.Frame(f, bg=color, height=3).pack(fill="x")
    for l in lines:
        tk.Label(f, text=l, bg=BG_CARD, fg=TEXT_MAIN,
                 font=FONT_CODE, anchor="w").pack(anchor="w", padx=8, pady=1)

def style_ax(ax, title, color):
    """Light theme for 2D axes."""
    ax.set_facecolor("#FAFAFA")
    ax.spines['bottom'].set_color(BORDER)
    ax.spines['left'].set_color(BORDER)
    ax.spines['top'].set_color(BORDER)
    ax.spines['right'].set_color(BORDER)
    ax.tick_params(colors=TEXT_SUB, labelsize=7)
    ax.set_title(title, color=color, fontsize=9, pad=8, fontweight='bold')
    ax.xaxis.label.set_color(TEXT_SUB)
    ax.yaxis.label.set_color(TEXT_SUB)

def style_ax3d(ax, title, color, xlabel="X1", ylabel="X2", zlabel="Z"):
    ax.set_facecolor("#FAFAFA")
    ax.set_xlabel(xlabel, color=TEXT_SUB, fontsize=8, labelpad=6)
    ax.set_ylabel(ylabel, color=TEXT_SUB, fontsize=8, labelpad=6)
    ax.set_zlabel(zlabel, color=TEXT_SUB, fontsize=8, labelpad=6)
    ax.set_title(title, color=color, fontsize=9, pad=8, fontweight='bold')
    ax.tick_params(colors=TEXT_SUB, labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor("#F0F4F8")
        pane.set_edgecolor(BORDER)
    ax.xaxis.line.set_color(BORDER)
    ax.yaxis.line.set_color(BORDER)
    ax.zaxis.line.set_color(BORDER)

def make_sidebar(win, title, icon, color):
    left = tk.Frame(win, bg=BG_PANEL, width=280)
    left.pack(side="left", fill="y", padx=0, pady=0)
    left.pack_propagate(False)
    # green top accent bar
    tk.Frame(left, bg=color, height=4).pack(fill="x")
    # header
    hdr = tk.Frame(left, bg=BG_PANEL)
    hdr.pack(fill="x", padx=16, pady=(14, 4))
    tk.Label(hdr, text=icon, bg=BG_PANEL, fg=color,
             font=("Segoe UI", 20)).pack(side="left")
    tk.Label(hdr, text=title, bg=BG_PANEL, fg=color,
             font=("Segoe UI", 12, "bold"), wraplength=200,
             justify="left").pack(side="left", padx=8)
    tk.Frame(left, bg=BORDER, height=1).pack(fill="x", padx=16, pady=6)
    return left

def make_canvas(win):
    right = tk.Frame(win, bg=BG_MAIN)
    right.pack(side="left", fill="both", expand=True, padx=12, pady=12)
    return right

# ═══════════════════════════════════════════════════════════════════════════════
#  1. RÉGRESSION LINÉAIRE 3D
# ═══════════════════════════════════════════════════════════════════════════════
def open_regression(root):
    win = tk.Toplevel(root)
    win.title("Régression Linéaire Multiple 3D")
    win.configure(bg=BG_MAIN)
    win.geometry("1100x700")

    left = make_sidebar(win, "Régression Linéaire 3D", "📈", ACCENT1)
    section_label(left, "Variable X1", ACCENT1)
    x1min = labeled_entry(left, "Minimum", -10, ACCENT1)
    x1max = labeled_entry(left, "Maximum", 10,  ACCENT1)
    section_label(left, "Variable X2", ACCENT1)
    x2min = labeled_entry(left, "Minimum", -5,  ACCENT1)
    x2max = labeled_entry(left, "Maximum", 15,  ACCENT1)
    section_label(left, "Dataset", ACCENT1)
    n_pts = labeled_entry(left, "Nb points", 200, ACCENT1)
    metrics_frame = tk.Frame(left, bg=BG_PANEL)
    metrics_frame.pack(fill="x", padx=6)

    right = make_canvas(win)
    fig = plt.Figure(figsize=(8, 5.5), facecolor=BG_CARD)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def run():
        for w in metrics_frame.winfo_children(): w.destroy()
        try:
            n  = int(n_pts.get())
            x1 = np.random.uniform(float(x1min.get()), float(x1max.get()), n)
            x2 = np.random.uniform(float(x2min.get()), float(x2max.get()), n)
            noise = np.random.normal(0, 2, n)
            a0, a1, a2 = 4.5, 2.1, -1.3
            y  = a0 + a1*x1 + a2*x2 + noise
            X  = np.column_stack([np.ones(n), x1, x2])
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            y_hat = X @ coef
            ss_res = np.sum((y - y_hat)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2   = 1 - ss_res/ss_tot
            mse  = ss_res/n
            rmse = np.sqrt(mse)

            fig.clf()
            fig.patch.set_facecolor(BG_CARD)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(x1, x2, y, c=y, cmap='RdYlGn', s=18, alpha=0.7)
            gx1, gx2 = np.meshgrid(np.linspace(x1.min(), x1.max(), 15),
                                    np.linspace(x2.min(), x2.max(), 15))
            gz = coef[0] + coef[1]*gx1 + coef[2]*gx2
            ax.plot_surface(gx1, gx2, gz, alpha=0.30, color=ACCENT1)
            style_ax3d(ax, "Régression Linéaire Multiple 3D", ACCENT1, "X1", "X2", "Y")
            cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.08)
            cb.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB)
            canvas.draw()

            metrics_box(metrics_frame, [
                "  Métriques de performance",
                "  ─────────────────────────",
                f"  R²   = {r2:.4f}",
                f"  MSE  = {mse:.4f}",
                f"  RMSE = {rmse:.4f}",
                "",
                "  Coefficients",
                f"  β₀ = {coef[0]:.4f}",
                f"  β₁ = {coef[1]:.4f}",
                f"  β₂ = {coef[2]:.4f}",
                "",
                "  Équation",
                f"  Y = {coef[0]:.2f} + {coef[1]:.2f}·X1 + {coef[2]:.2f}·X2",
            ], ACCENT1)
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex))

    run_button(left, "▶  Lancer la Régression", run, ACCENT1)
    run()

# ═══════════════════════════════════════════════════════════════════════════════
#  2. CLUSTERING K-MEANS 3D
# ═══════════════════════════════════════════════════════════════════════════════
def open_clustering(root):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    win = tk.Toplevel(root)
    win.title("Clustering K-Means 3D")
    win.configure(bg=BG_MAIN)
    win.geometry("1350x760")

    left = make_sidebar(win, "Clustering K-Means 3D", "📊", ACCENT2)
    scroll = ScrollFrame(left)
    scroll.pack(fill="both", expand=True)
    left = scroll.inner
    section_label(left, "Variables", ACCENT2)
    x1min = labeled_entry(left, "X1 min", -10, ACCENT2)
    x1max = labeled_entry(left, "X1 max", 10,  ACCENT2)
    x2min = labeled_entry(left, "X2 min", -5,  ACCENT2)
    x2max = labeled_entry(left, "X2 max", 15,  ACCENT2)
    x3min = labeled_entry(left, "X3 min", 0,   ACCENT2)
    x3max = labeled_entry(left, "X3 max", 20,  ACCENT2)
    section_label(left, "Modèle", ACCENT2)
    k_var = labeled_entry(left, "Nb clusters", 3,   ACCENT2)
    n_var = labeled_entry(left, "Nb échantillons", 300, ACCENT2)
    metrics_frame = tk.Frame(left, bg=BG_PANEL)
    metrics_frame.pack(fill="x", padx=6)

    right = make_canvas(win)
    fig = plt.Figure(figsize=(11, 7), facecolor=BG_CARD)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    CMAP = ['#00843D','#0072C6','#E63946','#8338EC','#F77F00']

    def run():
        for w in metrics_frame.winfo_children(): w.destroy()
        try:
            n = int(n_var.get()); k = int(k_var.get())
            x1 = np.random.uniform(float(x1min.get()), float(x1max.get()), n)
            x2 = np.random.uniform(float(x2min.get()), float(x2max.get()), n)
            x3 = np.random.uniform(float(x3min.get()), float(x3max.get()), n)
            X  = np.column_stack([x1, x2, x3])
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)

            fig.clf(); fig.patch.set_facecolor(BG_CARD)
            titles  = ["X1 – X2 – X3", "X1 – X3 – X2", "X2 – X3 – X1"]
            combos  = [(0,1,2), (0,2,1), (1,2,0)]
            xlabels = ["X1","X1","X2"]; ylabels = ["X2","X3","X3"]; zlabels = ["X3","X2","X1"]

            for col, (title, (ix, iy, iz), xl, yl, zl) in enumerate(
                    zip(titles, combos, xlabels, ylabels, zlabels)):
                ax = fig.add_subplot(1, 3, col+1, projection='3d')
                for c in range(k):
                    mask = labels == c
                    ax.scatter(X[mask,ix], X[mask,iy], X[mask,iz],
                               s=22, color=CMAP[c % len(CMAP)], alpha=0.75,
                               label=f"C{c}", depthshade=True)
                cx = km.cluster_centers_
                ax.scatter(cx[:,ix], cx[:,iy], cx[:,iz],
                           marker='*', s=300, color='#FFD700',
                           zorder=8, edgecolors='#333', linewidths=0.6, label="Centres")
                style_ax3d(ax, title, ACCENT2, xl, yl, zl)
                if col == 0:
                    ax.legend(facecolor=BG_CARD, labelcolor=TEXT_MAIN, fontsize=6, loc='upper left')

            fig.tight_layout(pad=2); canvas.draw()

            sizes = [np.sum(labels == c) for c in range(k)]
            lines = ["  Résultats du clustering","  ─────────────────────────",
                     f"  Nb clusters : {k}", f"  Silhouette  : {sil:.4f}", "","  Centres des clusters",
            ] + [f"  C{i}: [{', '.join([f'{v:.2f}' for v in km.cluster_centers_[i]])}]" for i in range(k)
            ] + ["","  Taille des clusters"
            ] + [f"  C{i}: {sizes[i]} pts ({sizes[i]/n*100:.1f}%)" for i in range(k)]
            metrics_box(metrics_frame, lines, ACCENT2)
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex))

    run_button(left, "▶  Lancer le Clustering", run, ACCENT2)
    run()

# ═══════════════════════════════════════════════════════════════════════════════
#  3. RANDOM FOREST 3D
# ═══════════════════════════════════════════════════════════════════════════════
def open_rf(root):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, f1_score,
                                  confusion_matrix, precision_score, recall_score)
    from sklearn.model_selection import train_test_split

    win = tk.Toplevel(root)
    win.title("Random Forest 3D")
    win.configure(bg=BG_MAIN)
    win.geometry("1200x720")

    left = make_sidebar(win, "Random Forest 3D", "🌳", ACCENT3)
    section_label(left, "Données", ACCENT3)
    x1min = labeled_entry(left, "X1 min", -10, ACCENT3)
    x1max = labeled_entry(left, "X1 max", 10,  ACCENT3)
    x2min = labeled_entry(left, "X2 min", -5,  ACCENT3)
    x2max = labeled_entry(left, "X2 max", 15,  ACCENT3)
    x3min = labeled_entry(left, "X3 min", 0,   ACCENT3)
    x3max = labeled_entry(left, "X3 max", 20,  ACCENT3)
    section_label(left, "Modèle", ACCENT3)
    n_trees = labeled_entry(left, "Nb arbres",  100, ACCENT3)
    n_cls   = labeled_entry(left, "Nb classes",   3, ACCENT3)
    n_samp  = labeled_entry(left, "Nb samples", 300, ACCENT3)
    metrics_frame = tk.Frame(left, bg=BG_PANEL)
    metrics_frame.pack(fill="x", padx=6)

    right = make_canvas(win)
    fig = plt.Figure(figsize=(11, 6), facecolor=BG_CARD)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    CMAP_CLF = ['#00843D','#0072C6','#E63946','#8338EC','#F77F00']

    def run():
        for w in metrics_frame.winfo_children(): w.destroy()
        try:
            n = int(n_samp.get()); nt = int(n_trees.get()); nc = int(n_cls.get())
            x1 = np.random.uniform(float(x1min.get()), float(x1max.get()), n)
            x2 = np.random.uniform(float(x2min.get()), float(x2max.get()), n)
            x3 = np.random.uniform(float(x3min.get()), float(x3max.get()), n)
            y  = (np.abs(x1+x2+x3) * np.random.rand(n)).astype(int) % nc
            X  = np.column_stack([x1, x2, x3])
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
            rf = RandomForestClassifier(n_estimators=nt, random_state=42)
            rf.fit(Xtr, ytr); yp = rf.predict(Xte)
            acc = accuracy_score(yte, yp)
            f1  = f1_score(yte, yp, average='macro', zero_division=0)
            prec= precision_score(yte, yp, average='macro', zero_division=0)
            rec = recall_score(yte, yp, average='macro', zero_division=0)
            cm  = confusion_matrix(yte, yp)
            imp = rf.feature_importances_

            fig.clf(); fig.patch.set_facecolor(BG_CARD)

            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            for xi, (h, c) in enumerate(zip(imp, [ACCENT3, ACCENT5, ACCENT4])):
                ax1.bar3d(xi-0.3, 0, 0, 0.6, 0.6, h, color=c, alpha=0.85)
            ax1.set_xticks([0,1,2]); ax1.set_xticklabels(['X1','X2','X3'], color=TEXT_SUB, fontsize=7)
            style_ax3d(ax1, "Importance des variables", ACCENT3, "Variable", "", "Importance")
            ax1.set_yticks([])

            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            _x, _y = np.meshgrid(range(nc), range(nc))
            _x = _x.ravel(); _y = _y.ravel()
            dz = cm.ravel()
            colors_cm = [CMAP_CLF[i % len(CMAP_CLF)] for i in _x]
            ax2.bar3d(_x-0.4, _y-0.4, np.zeros_like(_x), 0.8, 0.8, dz, color=colors_cm, alpha=0.85, shade=True)
            ax2.set_xticks(range(nc)); ax2.set_yticks(range(nc))
            ax2.set_xticklabels([f"P{i}" for i in range(nc)], color=TEXT_SUB, fontsize=7)
            ax2.set_yticklabels([f"R{i}" for i in range(nc)], color=TEXT_SUB, fontsize=7)
            style_ax3d(ax2, "Matrice de confusion 3D", ACCENT3, "Prédit", "Réel", "Count")

            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            for c in range(nc):
                mask = yte == c
                ax3.scatter(Xte[mask,0], Xte[mask,1], Xte[mask,2],
                            s=20, color=CMAP_CLF[c % len(CMAP_CLF)],
                            alpha=0.80, label=f"Cl {c}", depthshade=True)
                wrong = (yte == c) & (yp != c)
                if wrong.any():
                    ax3.scatter(Xte[wrong,0], Xte[wrong,1], Xte[wrong,2],
                                s=40, marker='x', color=TEXT_MAIN, alpha=0.9, linewidths=1)
            style_ax3d(ax3, "Scatter 3D (× = erreurs)", ACCENT3, "X1", "X2", "X3")
            ax3.legend(facecolor=BG_CARD, labelcolor=TEXT_MAIN, fontsize=6, loc='upper left')

            fig.tight_layout(pad=2); canvas.draw()
            metrics_box(metrics_frame, [
                "  Métriques de performance","  ─────────────────────────",
                f"  Accuracy  : {acc:.4f}", f"  Précision : {prec:.4f}",
                f"  Rappel    : {rec:.4f}", f"  Score F1  : {f1:.4f}",
                "","  Importance des variables",
                f"  X1: {imp[0]*100:.2f}%", f"  X2: {imp[1]*100:.2f}%", f"  X3: {imp[2]*100:.2f}%",
            ], ACCENT3)
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex))

    run_button(left, "▶  Lancer Random Forest", run, ACCENT3)
    run()

# ═══════════════════════════════════════════════════════════════════════════════
#  4. TIME SERIES ARIMA 3D
# ═══════════════════════════════════════════════════════════════════════════════
def open_timeseries(root):
    win = tk.Toplevel(root)
    win.title("Time Series ARIMA 3D")
    win.configure(bg=BG_MAIN)
    win.geometry("1100x680")

    left = make_sidebar(win, "Time Series ARIMA 3D", "⏱", ACCENT4)
    section_label(left, "Paramètres ARIMA", ACCENT4)
    p_var  = labeled_entry(left, "p (AR)",         1,   ACCENT4)
    d_var  = labeled_entry(left, "d (Différence)", 1,   ACCENT4)
    q_var  = labeled_entry(left, "q (MA)",         1,   ACCENT4)
    section_label(left, "Données", ACCENT4)
    n_var  = labeled_entry(left, "Nb points",      200, ACCENT4)
    fh_var = labeled_entry(left, "Horizon prévis", 30,  ACCENT4)
    metrics_frame = tk.Frame(left, bg=BG_PANEL)
    metrics_frame.pack(fill="x", padx=6)

    right = make_canvas(win)
    fig = plt.Figure(figsize=(8.5, 5.5), facecolor=BG_CARD)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def run():
        for w in metrics_frame.winfo_children(): w.destroy()
        try:
            from statsmodels.tsa.arima.model import ARIMA
            n = int(n_var.get()); fh = int(fh_var.get())
            p = int(p_var.get()); d = int(d_var.get()); q = int(q_var.get())
            t = np.arange(n)
            series = np.cumsum(np.random.normal(0,1.5,n)) + 0.05*t + 5*np.sin(0.2*t)
            split = int(n * 0.8)
            train = series[:split]; test = series[split:]
            model  = ARIMA(train, order=(p, d, q))
            result = model.fit()
            pred_test = result.forecast(steps=len(test))
            forecast  = result.forecast(steps=fh)
            mse  = np.mean((test - pred_test)**2); rmse = np.sqrt(mse)

            fig.clf(); fig.patch.set_facecolor(BG_CARD)
            ax = fig.add_subplot(111, projection='3d')

            def ribbon(ax, t_arr, z_arr, y_pos, color, label, lw=1.5, alpha=0.40):
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                verts = []
                for i in range(len(t_arr)-1):
                    xs = [t_arr[i], t_arr[i+1], t_arr[i+1], t_arr[i]]
                    ys = [y_pos, y_pos, y_pos, y_pos]
                    zs = [z_arr[i], z_arr[i+1], 0, 0]
                    verts.append(list(zip(xs, ys, zs)))
                poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='none')
                ax.add_collection3d(poly)
                ax.plot(t_arr, [y_pos]*len(t_arr), z_arr, color=color, lw=lw, label=label)

            ribbon(ax, t[:split], train, 0, ACCENT4, "Train")
            ribbon(ax, t[split:], test, 0, TEXT_SUB, "Test réel", alpha=0.25)
            ribbon(ax, t[split:], pred_test, 1, ACCENT5, "Prédit (test)")
            future_t = np.arange(n, n+fh)
            ribbon(ax, future_t, forecast, 2, ACCENT1, "Prévision future")
            lo = forecast - 1.96*rmse; hi = forecast + 1.96*rmse
            ax.plot_surface(np.array([future_t, future_t]),
                            np.array([[2]*fh, [2]*fh]),
                            np.array([lo, hi]), alpha=0.15, color=ACCENT1)
            z_range = np.linspace(series.min()-2, series.max()+2, 5)
            ax.plot([split]*5, [0]*5, z_range, color=BORDER, lw=1, linestyle=':')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["Données", "Prédit", "Futur"], color=TEXT_SUB, fontsize=7)
            style_ax3d(ax, f"ARIMA({p},{d},{q}) – Ribbon 3D", ACCENT4, "Temps", "", "Valeur")
            ax.legend(facecolor=BG_CARD, labelcolor=TEXT_MAIN, fontsize=7, loc='upper left')
            fig.tight_layout(pad=2); canvas.draw()

            metrics_box(metrics_frame, [
                f"  Modèle ARIMA({p},{d},{q})","  ─────────────────────────",
                f"  AIC  : {result.aic:.2f}", f"  BIC  : {result.bic:.2f}",
                f"  MSE  : {mse:.4f}", f"  RMSE : {rmse:.4f}",
                "  Prévision à +{fh} pas", f"  μ prévision : {forecast.mean():.4f}",
            ], ACCENT4)
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex))

    run_button(left, "▶  Lancer ARIMA", run, ACCENT4)
    run()

# ═══════════════════════════════════════════════════════════════════════════════
#  5. RÉSEAUX DE NEURONES 3D
# ═══════════════════════════════════════════════════════════════════════════════
def open_nn(root):
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split

    win = tk.Toplevel(root)
    win.title("Réseaux de Neurones 3D")
    win.configure(bg=BG_MAIN)
    win.geometry("1150x700")

    left = make_sidebar(win, "Réseaux de Neurones 3D", "🧠", ACCENT5)
    section_label(left, "Architecture", ACCENT5)
    h1_var = labeled_entry(left, "Couche 1 (neurones)", 64,    ACCENT5)
    h2_var = labeled_entry(left, "Couche 2 (neurones)", 32,    ACCENT5)
    lr_var = labeled_entry(left, "Learning rate",       0.001, ACCENT5)
    it_var = labeled_entry(left, "Max itérations",      500,   ACCENT5)
    section_label(left, "Données", ACCENT5)
    n_var  = labeled_entry(left, "Nb points", 300, ACCENT5)
    metrics_frame = tk.Frame(left, bg=BG_PANEL)
    metrics_frame.pack(fill="x", padx=6)

    right = make_canvas(win)
    fig = plt.Figure(figsize=(9.5, 5.5), facecolor=BG_CARD)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def run():
        for w in metrics_frame.winfo_children(): w.destroy()
        try:
            n  = int(n_var.get())
            h1 = int(h1_var.get()); h2 = int(h2_var.get())
            lr = float(lr_var.get()); itr = int(it_var.get())
            X  = np.random.uniform(-5, 5, (n, 3))
            y  = np.sin(X[:,0]) + 0.5*X[:,1]**2 - X[:,2] + np.random.normal(0,0.3,n)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            nn = MLPRegressor(hidden_layer_sizes=(h1, h2), max_iter=itr,
                              learning_rate_init=lr, random_state=42,
                              early_stopping=True, validation_fraction=0.1)
            nn.fit(Xtr, ytr)
            yp   = nn.predict(Xte)
            mse  = np.mean((yte-yp)**2); rmse = np.sqrt(mse)
            mae  = np.mean(np.abs(yte-yp))
            r2   = 1 - np.sum((yte-yp)**2)/np.sum((yte-yte.mean())**2)

            fig.clf(); fig.patch.set_facecolor(BG_CARD)

            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            loss_curve = np.array(nn.loss_curve_)
            n_ep = len(loss_curve); n_runs = 8
            eps = np.arange(n_ep); runs = np.arange(n_runs)
            EPS, RUNS = np.meshgrid(eps, runs)
            LOSS = np.array([
                loss_curve + np.random.normal(0, loss_curve.std()*0.15, n_ep) * (1 + 0.1*r)
                for r in range(n_runs)
            ])
            ax1.plot_surface(EPS, RUNS, LOSS, cmap='YlGn', alpha=0.82, shade=True)
            ax1.plot(eps, [0]*n_ep, loss_curve, color=ACCENT5, lw=2, zorder=5, label="Run réel")
            style_ax3d(ax1, "Surface de perte 3D", ACCENT5, "Époques", "Runs", "Loss")
            ax1.legend(facecolor=BG_CARD, labelcolor=TEXT_MAIN, fontsize=7)

            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            err = np.abs(yte - yp)
            sc  = ax2.scatter(Xte[:,0], Xte[:,1], yte, c=err, cmap='RdYlGn_r', s=20, alpha=0.80, depthshade=True)
            for i in range(len(yte)):
                ax2.plot([Xte[i,0], Xte[i,0]], [Xte[i,1], Xte[i,1]], [yte[i], yp[i]],
                         color=ACCENT3, alpha=0.20, lw=0.6)
            ax2.scatter(Xte[:,0], Xte[:,1], yp, color=ACCENT5, s=14, alpha=0.6, marker='^', label="Prédit")
            fig.colorbar(sc, ax=ax2, shrink=0.45, pad=0.1, label="Erreur abs.").ax.yaxis.set_tick_params(
                color=TEXT_SUB, labelcolor=TEXT_SUB)
            style_ax3d(ax2, "Réel vs Prédit – Scatter 3D", ACCENT5, "X1", "X2", "Valeur Y")
            ax2.legend(facecolor=BG_CARD, labelcolor=TEXT_MAIN, fontsize=7)
            fig.tight_layout(pad=2); canvas.draw()

            metrics_box(metrics_frame, [
                "  Métriques de performance","  ─────────────────────────",
                f"  R²   : {r2:.4f}", f"  RMSE : {rmse:.4f}", f"  MAE  : {mae:.4f}", f"  MSE  : {mse:.4f}",
                "","  Architecture",
                f"  Input → {h1} → {h2} → Output", f"  LR   : {lr}", f"  Iter : {nn.n_iter_}",
            ], ACCENT5)
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex))

    run_button(left, "▶  Lancer le Réseau", run, ACCENT5)
    run()

# ═══════════════════════════════════════════════════════════════════════════════
#  6. VALIDATION CROISÉE 3D
# ═══════════════════════════════════════════════════════════════════════════════
def open_cv(root):
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    import time

    win = tk.Toplevel(root)
    win.title("Validation Croisée 3D")
    win.configure(bg=BG_MAIN)
    win.geometry("1250x720")

    left = make_sidebar(win, "Validation Croisée 3D", "🔁", ACCENT6)
    section_label(left, "Paramètres", ACCENT6)
    k_var = labeled_entry(left, "Nb folds",   5,   ACCENT6)
    n_var = labeled_entry(left, "Nb samples", 300, ACCENT6)
    metrics_frame = tk.Frame(left, bg=BG_PANEL)
    metrics_frame.pack(fill="x", padx=6)

    right = make_canvas(win)
    fig = plt.Figure(figsize=(11, 6.5), facecolor=BG_CARD)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    colors_m = [ACCENT3, ACCENT5, ACCENT2, ACCENT4]
    model_names = ["Random Forest", "Réseau Neurones", "SVM", "Arbre Décision"]
    short_names = ["RF", "NN", "SVM", "DT"]

    def run():
        for w in metrics_frame.winfo_children(): w.destroy()
        try:
            n = int(n_var.get()); k = int(k_var.get())
            X = np.random.randn(n, 3)
            y2 = np.where(X[:,0]>0, np.where(X[:,1]>0, 2, 1), 0)
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            models_dict = {
                "Random Forest":   RandomForestClassifier(n_estimators=50, random_state=42),
                "Réseau Neurones": MLPClassifier(max_iter=300, random_state=42),
                "SVM":             SVC(random_state=42),
                "Arbre Décision":  DecisionTreeClassifier(random_state=42),
            }
            results_acc = {}; results_f1 = {}; t_train = []
            for name, clf in models_dict.items():
                results_acc[name] = cross_val_score(clf, X, y2, cv=cv, scoring='accuracy')
                results_f1[name]  = cross_val_score(clf, X, y2, cv=cv, scoring='f1_macro')
                t0 = time.time(); clf.fit(X, y2); t_train.append((time.time()-t0)*1000)

            nm = len(model_names)
            fig.clf(); fig.patch.set_facecolor(BG_CARD)

            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            folds = np.arange(1, k+1)
            for mi, (name, color) in enumerate(zip(model_names, colors_m)):
                scores = results_acc[name]
                ax1.bar3d(folds-0.35, [mi]*k, np.zeros(k), 0.7, 0.7, scores, color=color, alpha=0.80, shade=True)
            ax1.set_xticks(folds); ax1.set_xticklabels([f"F{f}" for f in folds], color=TEXT_SUB, fontsize=7)
            ax1.set_yticks(range(nm)); ax1.set_yticklabels(short_names, color=TEXT_SUB, fontsize=7)
            style_ax3d(ax1, "Accuracy par fold 3D", ACCENT6, "Fold", "Modèle", "Accuracy")

            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            F1_mat = np.array([results_f1[n] for n in model_names])
            FOLDS_grid, MODELS_grid = np.meshgrid(np.arange(k), np.arange(nm))
            ax2.plot_surface(FOLDS_grid, MODELS_grid, F1_mat, cmap='cool', alpha=0.85, shade=True)
            for mi, (name, color) in enumerate(zip(model_names, colors_m)):
                ax2.plot(np.arange(k), [mi]*k, results_f1[name], color=color, lw=2, zorder=5)
            ax2.set_xticks(range(k)); ax2.set_xticklabels([f"F{f+1}" for f in range(k)], color=TEXT_SUB, fontsize=7)
            ax2.set_yticks(range(nm)); ax2.set_yticklabels(short_names, color=TEXT_SUB, fontsize=7)
            style_ax3d(ax2, "Surface F1 Score 3D", ACCENT6, "Fold", "Modèle", "F1")

            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            for mi, (t, c) in enumerate(zip(t_train, colors_m)):
                ax3.bar3d(mi-0.35, 0, 0, 0.7, 0.5, t, color=c, alpha=0.85)
            ax3.set_xticks(range(nm)); ax3.set_xticklabels(short_names, color=TEXT_SUB, fontsize=7)
            ax3.set_yticks([])
            style_ax3d(ax3, "Temps d'entraînement 3D", ACCENT6, "Modèle", "", "ms")

            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            for mi, (name, color) in enumerate(zip(model_names, colors_m)):
                scores = results_acc[name]
                t_arr = np.arange(1, k+1, dtype=float)
                verts = []
                for i in range(len(t_arr)-1):
                    xs = [t_arr[i], t_arr[i+1], t_arr[i+1], t_arr[i]]
                    ys = [mi, mi, mi, mi]
                    zs = [scores[i], scores[i+1], 0, 0]
                    verts.append(list(zip(xs, ys, zs)))
                poly = Poly3DCollection(verts, alpha=0.4, facecolor=color, edgecolor='none')
                ax4.add_collection3d(poly)
                ax4.plot(t_arr, [mi]*k, scores, color=color, lw=2, label=short_names[mi])
            ax4.set_xticks(np.arange(1, k+1)); ax4.set_xticklabels([f"F{f}" for f in range(1, k+1)], color=TEXT_SUB, fontsize=7)
            ax4.set_yticks(range(nm)); ax4.set_yticklabels(short_names, color=TEXT_SUB, fontsize=7)
            style_ax3d(ax4, "Ribbon Accuracy 3D", ACCENT6, "Fold", "Modèle", "Accuracy")
            ax4.legend(facecolor=BG_CARD, labelcolor=TEXT_MAIN, fontsize=6, loc='upper left')

            fig.tight_layout(pad=2); canvas.draw()

            means = [results_acc[n].mean() for n in model_names]
            lines = ["  Résultats validation croisée","  ─────────────────────────────"]
            for name in model_names:
                lines += [f"  {name}:", f"    Acc: {results_acc[name].mean():.4f} ± {results_acc[name].std():.4f}"]
            lines += ["", f"  Meilleur modèle : {model_names[np.argmax(means)]}"]
            metrics_box(metrics_frame, lines, ACCENT6)
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex))

    run_button(left, "▶  Lancer la Validation", run, ACCENT6)
    run()

# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHMS SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
def open_algos(root):
    win = tk.Toplevel(root)
    win.title("Algorithmes de l'IA – EMSI")
    win.configure(bg=BG_MAIN)
    win.geometry("860x560")
    win.resizable(False, False)

    # ── top accent stripe ──
    stripe = tk.Frame(win, bg=EMSI_GREEN, height=4)
    stripe.pack(fill="x")

    # ── header with EMSI logo area ──
    header = tk.Frame(win, bg=BG_CARD,
                      highlightthickness=1, highlightbackground=BORDER)
    header.pack(fill="x", padx=20, pady=(14, 0))
    tk.Frame(header, bg=EMSI_GREEN, width=6).pack(side="left", fill="y")

    # EMSI logo placeholder (text-based logo)
    logo_f = tk.Frame(header, bg=BG_CARD, padx=14, pady=10)
    logo_f.pack(side="left")
    try:
        img_raw = Image.open(r"C:\Users\Hp\OneDrive\Images\Saved Pictures\OIP.webp").resize((180, 60), Image.LANCZOS)
        logo_img = ImageTk.PhotoImage(img_raw)
        lbl = tk.Label(logo_f, image=logo_img, bg=BG_CARD)
        lbl.image = logo_img
        lbl.pack()
    except:
        tk.Label(logo_f, text="EMSI", bg=EMSI_GREEN, fg="white",
                 font=("Segoe UI", 16, "bold"), padx=10, pady=6).pack()

    title_f = tk.Frame(header, bg=BG_CARD, padx=16, pady=10)
    title_f.pack(side="left", fill="both", expand=True)
    tk.Label(title_f, text="Algorithmes de l'Intelligence Artificielle",
             bg=BG_CARD, fg=TEXT_MAIN,
             font=("Segoe UI", 14, "bold")).pack(anchor="w")
    tk.Label(title_f, text="Machine Learning  ·  3ème Année Informatique",
             bg=BG_CARD, fg=TEXT_SUB, font=("Segoe UI", 9)).pack(anchor="w")

    # ── grid ──
    outer = tk.Frame(win, bg=BORDER, bd=1)
    outer.pack(fill="both", expand=True, padx=20, pady=14)
    grid = tk.Frame(outer, bg=BG_MAIN)
    grid.pack(fill="both", expand=True, padx=2, pady=2)

    algo_actions = {
        "Régression Linéaire": lambda: open_regression(root),
        "Clustering":          lambda: open_clustering(root),
        "Random Forest":       lambda: open_rf(root),
        "Time Series ARIMA":   lambda: open_timeseries(root),
        "Réseaux de Neurones": lambda: open_nn(root),
        "Validation Croisée":  lambda: open_cv(root),
    }
    ICONS = {
        "Régression Linéaire": "📈",
        "Clustering":          "📊",
        "Random Forest":       "🌳",
        "Time Series ARIMA":   "⏱",
        "Réseaux de Neurones": "🧠",
        "Validation Croisée":  "🔁",
    }
    DESCS = {
        "Régression Linéaire": "Modéliser une relation linéaire",
        "Clustering":          "Regrouper données similaires",
        "Random Forest":       "Prédiction par ensemble d'arbres",
        "Time Series ARIMA":   "Analyser & prédire dans le temps",
        "Réseaux de Neurones": "Modèles non-linéaires complexes",
        "Validation Croisée":  "Comparer et valider les modèles",
    }

    for idx, (name, action) in enumerate(algo_actions.items()):
        row, col = divmod(idx, 3)
        color = ALGO_COLORS[name]
        cell = tk.Frame(grid, bg=BG_CARD,
                        highlightthickness=1, highlightbackground=BORDER)
        cell.grid(row=row, column=col, padx=9, pady=9,
                  sticky="nsew", ipadx=10, ipady=8)

        # colored top bar per card
        tk.Frame(cell, bg=color, height=3).pack(fill="x")

        def make_enter(c, clr):
            def enter(e):
                c.config(highlightbackground=clr, bg=BTN_HOVER)
                for child in c.winfo_children():
                    try: child.config(bg=BTN_HOVER)
                    except: pass
            return enter
        def make_leave(c):
            def leave(e):
                c.config(highlightbackground=BORDER, bg=BG_CARD)
                for child in c.winfo_children():
                    try: child.config(bg=BG_CARD)
                    except: pass
            return leave

        cell.bind("<Enter>", make_enter(cell, color))
        cell.bind("<Leave>", make_leave(cell))

        icon_lbl = tk.Label(cell, text=ICONS[name], bg=BG_CARD, fg=color,
                            font=("Segoe UI", 22))
        icon_lbl.pack(pady=(10, 2))
        btn = tk.Button(cell, text=name, bg=BG_CARD, fg=color,
                        font=("Segoe UI", 10, "bold"),
                        relief="flat", cursor="hand2",
                        activebackground=BTN_HOVER, activeforeground=EMSI_DGRN,
                        command=action, wraplength=160)
        btn.pack(pady=(0, 2))
        desc_lbl = tk.Label(cell, text=DESCS[name], bg=BG_CARD, fg=TEXT_SUB,
                            font=("Segoe UI", 8), wraplength=160)
        desc_lbl.pack(pady=(0, 8))
        btn.bind("<Enter>", make_enter(cell, color))
        btn.bind("<Leave>", make_leave(cell))
        icon_lbl.bind("<Button-1>", lambda e, a=action: a())

    for c in range(3):
        grid.columnconfigure(c, weight=1, uniform="col")
    grid.rowconfigure(0, weight=1)
    grid.rowconfigure(1, weight=1)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    root.title("Application IA – EMSI Machine Learning")
    root.configure(bg=BG_MAIN)
    root.geometry("760x540")
    root.resizable(False, False)

    # ── top accent bar (green stripe) ──
    tk.Frame(root, bg=EMSI_GREEN, height=5).pack(fill="x")

    # ── main card ──
    outer = tk.Frame(root, bg=BORDER, bd=1)
    outer.pack(fill="both", expand=True, padx=28, pady=(20, 28))
    inner = tk.Frame(outer, bg=BG_CARD, padx=0, pady=0)
    inner.pack(fill="both", expand=True)

    # ── logo + title header ──
    header = tk.Frame(inner, bg=EMSI_GREEN)
    header.pack(fill="x")

    # Left: EMSI logo block
    logo_block = tk.Frame(header, bg="white", padx=20, pady=18)
    logo_block.pack(side="left")
    try:
        img_raw2 = Image.open(r"C:\Users\Hp\OneDrive\Images\Saved Pictures\OIP.webp").resize((200, 65), Image.LANCZOS)
        logo_img2 = ImageTk.PhotoImage(img_raw2)
        lbl2 = tk.Label(logo_block, image=logo_img2, bg="white")
        lbl2.image = logo_img2
        lbl2.pack()
    except:
        tk.Label(logo_block, text="EMSI", bg=EMSI_DGRN, fg="white",
                 font=("Segoe UI", 26, "bold")).pack()
    
    # Right: app title on green
    title_block = tk.Frame(header, bg=EMSI_GREEN, padx=24, pady=16)
    title_block.pack(side="left", fill="both", expand=True)
    tk.Label(title_block,
             text="IA & Machine Learning",
             bg=EMSI_GREEN, fg="white",
             font=("Segoe UI", 16, "bold")).pack(anchor="w")
    tk.Label(title_block,
             text="Application Desktop  ·  3ème Année Informatique",
             bg=EMSI_GREEN, fg="#C8E6C9",
             font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))

    # ── body ──
    body = tk.Frame(inner, bg=BG_CARD, padx=30, pady=20)
    body.pack(fill="both", expand=True)

    # info row
    info_row = tk.Frame(body, bg=EMSI_LGRN,
                        highlightthickness=1, highlightbackground=EMSI_GREEN)
    info_row.pack(fill="x", pady=(0, 20))
    tk.Frame(info_row, bg=EMSI_GREEN, width=4).pack(side="left", fill="y")
    tk.Label(info_row,
             text="  Algorithmes de l'Intelligence Artificielle  —  Visualisations 3D interactives",
             bg=EMSI_LGRN, fg=EMSI_DGRN,
             font=("Segoe UI", 10, "bold"), pady=10).pack(side="left")

    # button row
    btn_row = tk.Frame(body, bg=BG_CARD)
    btn_row.pack()

    def make_btn(parent, text, cmd, color, outline=False):
        if outline:
            f = tk.Frame(parent, bg=color, padx=1, pady=1)
            f.pack(side="left", padx=14)
            b = tk.Button(f, text=text, command=cmd,
                          bg=BG_CARD, fg=color,
                          font=FONT_BTN, relief="flat", cursor="hand2",
                          width=22, pady=10,
                          activebackground=color, activeforeground="white")
            b.pack()
            b.bind("<Enter>", lambda e: b.config(bg=color, fg="white"))
            b.bind("<Leave>", lambda e: b.config(bg=BG_CARD, fg=color))
        else:
            f = tk.Frame(parent, bg=color, padx=0, pady=0)
            f.pack(side="left", padx=14)
            b = tk.Button(f, text=text, command=cmd,
                          bg=color, fg="white",
                          font=FONT_BTN, relief="flat", cursor="hand2",
                          width=22, pady=10,
                          activebackground=EMSI_DGRN, activeforeground="white")
            b.pack(padx=1, pady=1)
            b.bind("<Enter>", lambda e: b.config(bg=EMSI_DGRN))
            b.bind("<Leave>", lambda e: b.config(bg=color))
        return b

    make_btn(btn_row, "  ▶   Algorithmes de l'IA",
             lambda: open_algos(root), EMSI_GREEN)
    make_btn(btn_row, "  ✕   Quitter l'application",
             lambda: root.destroy(), ACCENT3, outline=True)

    # ── footer ──
    tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=20)
    footer = tk.Frame(inner, bg=BG_CARD, pady=10)
    footer.pack(fill="x")
    tk.Label(footer,
             text="Dr. EL MKHALET MOUNA  ·  EMSI  ·  3ème Année Informatique  ·  Hami Ismail",
             bg=BG_CARD, fg=TEXT_SUB,
             font=("Segoe UI", 8)).pack()

    # ── bottom accent bar ──
    bot = tk.Frame(root, bg=EMSI_GREEN, height=5)
    bot.pack(fill="x", side="bottom")

    root.mainloop()

if __name__ == "__main__":
    main()