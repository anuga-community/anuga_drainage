"""Tkinter hydrograph viewer for anuga_drainage / Simple_SW_Inlets CSV logs.

Scans a folder for per-inlet hydrograph CSVs (the schema written by
``HydrographLogger`` -- a superset of the Simple_SW_Inlets columns) and renders,
per inlet, four diagnostic subplots; a View menu also shows a folder-combined
hydrograph. HiDPI-aware (reads ``Xft.dpi``) with live UI / plot font sliders.

Launch with the ``anuga-drainage-viewer`` console script or
``python -m anuga_drainage.viewer``.
"""
import os
import json
import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkfont
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def combine_hydrographs(csv_paths):
    """Sum Captured/Bypass across per-inlet hydrograph CSVs onto a common time axis.

    Outer-merges each CSV's ``Time_s``/``Captured_Q_cms``/``Bypass_Q_cms`` on
    ``Time_s`` and returns ``(combined, skipped)``: ``combined`` is a DataFrame
    with ``Time_s``, the per-step totals (``Captured_total_cms`` /
    ``Bypass_total_cms`` / ``Combined_total_cms``) and their cumulative volumes
    (``*_cum_m3``); ``skipped`` lists files lacking the required columns.
    ``combined`` is empty when no file qualifies. Pure (no GUI) -- unit tested.
    """
    cols = ["Time_s", "Captured_total_cms", "Bypass_total_cms", "Combined_total_cms",
            "Captured_cum_m3", "Bypass_cum_m3", "Combined_cum_m3"]
    merged = None
    skipped = []
    for fp in csv_paths:
        try:
            df = pd.read_csv(fp, usecols=["Time_s", "Captured_Q_cms", "Bypass_Q_cms"])
        except (ValueError, OSError):
            skipped.append(os.path.basename(fp))   # missing columns / unreadable
            continue
        base = os.path.basename(fp)
        df = df.rename(columns={"Captured_Q_cms": f"Captured__{base}",
                                "Bypass_Q_cms": f"Bypass__{base}"})
        merged = df if merged is None else pd.merge(merged, df, on="Time_s", how="outer")

    if merged is None:
        return pd.DataFrame(columns=cols), skipped

    merged = merged.sort_values("Time_s").reset_index(drop=True)
    cap_cols = [c for c in merged.columns if c.startswith("Captured__")]
    byp_cols = [c for c in merged.columns if c.startswith("Bypass__")]
    merged[cap_cols] = merged[cap_cols].fillna(0.0)
    merged[byp_cols] = merged[byp_cols].fillna(0.0)

    captured = merged[cap_cols].sum(axis=1)
    bypass = merged[byp_cols].sum(axis=1)
    combined = captured + bypass
    dt = merged["Time_s"].astype(float).diff().fillna(0.0)
    return pd.DataFrame({
        "Time_s": merged["Time_s"],
        "Captured_total_cms": captured,
        "Bypass_total_cms": bypass,
        "Combined_total_cms": combined,
        "Captured_cum_m3": (captured * dt).cumsum(),
        "Bypass_cum_m3": (bypass * dt).cumsum(),
        "Combined_cum_m3": (combined * dt).cumsum(),
    }), skipped


class HydrographViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANUGA Drainage — Hydrograph Viewer")

        # Saved per-machine preferences (font scales, window geometry) override the auto defaults.
        cfg = self._load_config()

        # Restore the saved window size/position, else size relative to the screen so it fits.
        if cfg.get("geometry"):
            self.root.geometry(cfg["geometry"])
        else:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            win_w = min(1200, int(screen_w * 0.9))
            win_h = min(950, int(screen_h * 0.9))
            self.root.geometry(f"{win_w}x{win_h}")

        self.selected_dir = tk.StringVar(value=os.getcwd())
        self.csv_files = []
        self.canvas = None
        self.toolbar = None
        self.standby_lbl = None

        # Currently displayed data, kept so the font slider can re-render in place.
        # current_view is "pit" (single-file 4-panel) or "combined" (folder totals),
        # so a font-slider change re-renders whichever view is showing.
        self.current_df = None
        self.current_title = None
        self.current_view = "pit"

        # --- Font scaling -----------------------------------------------------
        # We DON'T trust winfo_fpixels('1i'): on Wayland/XWayland it reports a bogus 96 DPI
        # (and winfo_screenmm* is back-filled to match, so the physical-size trick lies too).
        # _detect_scale() instead reads the user's configured Xft.dpi, then clamps.
        # Whatever the auto guess, both UI and plot fonts have live sliders so any machine tunes.
        # Saved slider values (per machine) win over the auto guess, so it self-configures after
        # the first manual adjustment.
        auto = self._detect_scale()
        # Plot font scale (matplotlib) and UI widget scale, each a live, user-tunable variable.
        self.font_scale = tk.DoubleVar(value=cfg.get("font_scale", auto))
        self.ui_scale = tk.DoubleVar(value=cfg.get("ui_scale", auto))

        # Shared Font objects: every widget references these, so reconfiguring them on a slider
        # move instantly resizes the whole UI. (base size at scale 1.0, bold?, monospace?)
        self._font_specs = {
            "header": (9, True, False),    # section labels
            "body":   (9, False, False),   # directory entry
            "button": (10, False, False),  # Browse / Refresh
            "mono":   (9, False, True),    # the CSV listbox
            "menu":   (11, True, False),   # the File menu bar + dropdown
            "standby": (9, False, False),  # placeholder message
        }
        self.ui_fonts = {}
        for key, (base, bold, mono) in self._font_specs.items():
            self.ui_fonts[key] = tkfont.Font(
                family="Courier" if mono else "Arial",
                size=base, weight="bold" if bold else "normal",
            )
        self._apply_ui_scale()

        # Bump the named default fonts too, so any Tk-drawn dialogs scale along with the app.
        for fname in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkFixedFont"):
            try:
                nf = tkfont.nametofont(fname)
                nf.configure(size=max(10, int(round(abs(nf.cget("size")) * self.ui_scale.get()))))
            except tk.TclError:
                pass

        # Route the window-manager close button (X) through our clean shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Add menu
        self.setup_menu()
        # Resizable split between the sidebar and the plot panel (drag the sash to resize)
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=8, sashrelief=tk.RAISED)
        self.paned.pack(fill=tk.BOTH, expand=True)
        self.setup_sidebar_controls()
        self.setup_graph_display_panel()

        # Auto-scan the current workspace folder on launch
        self.refresh_workspace_files()

    _CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".anuga_drainage_viewer.json")

    def _load_config(self):
        """Loads saved per-machine font-scale preferences; returns {} if none/invalid."""
        try:
            with open(self._CONFIG_PATH) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_config(self):
        """Persists the slider values and window geometry so the app self-configures next launch."""
        data = {"ui_scale": round(self.ui_scale.get(), 2),
                "font_scale": round(self.font_scale.get(), 2)}
        try:
            data["geometry"] = self.root.winfo_geometry()
        except tk.TclError:
            pass
        try:
            with open(self._CONFIG_PATH, "w") as f:
                json.dump(data, f)
        except OSError:
            pass

    @staticmethod
    def _xft_dpi():
        """The user's *configured* DPI from X resources (Xft.dpi), or None.

        On Wayland/GNOME, Tk runs through XWayland which always reports 96 DPI to
        legacy X apps (and back-fills winfo_screenmm* to match, so those lie too).
        Xft.dpi is the value GTK/Qt actually honour for HiDPI scaling, e.g. 192 -> 2x.
        """
        import re, subprocess
        try:
            out = subprocess.run(["xrdb", "-query"], capture_output=True,
                                 text=True, timeout=2).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        m = re.search(r"^Xft\.dpi:\s*([0-9.]+)", out, re.M)
        return float(m.group(1)) if m else None

    def _detect_dpi(self):
        """Best-effort *true* screen DPI, working around XWayland's bogus 96.

        Order: the user's configured Xft.dpi (preferred — it reflects intended
        scaling, not raw panel density), then env hints, then Tk's own numbers.
        """
        # 1. Xft.dpi — the user's chosen scaling; the one source that's reliably honest here.
        dpi = self._xft_dpi()
        if dpi and dpi > 0:
            return dpi
        # 2. Toolkit env overrides some HiDPI setups export.
        for var in ("QT_FONT_DPI",):
            try:
                v = float(os.environ.get(var, ""))
                if v > 0:
                    return v
            except ValueError:
                pass
        # 3. Fall back to Tk's physical-size / reported DPI (both often bogus, but clamped below).
        try:
            px = self.root.winfo_screenwidth()
            mm = self.root.winfo_screenmmwidth()
            tk_dpi = (px * 25.4 / mm) if mm and mm > 0 else 96.0
        except tk.TclError:
            tk_dpi = 96.0
        try:
            tk_dpi = max(tk_dpi, self.root.winfo_fpixels("1i"))
        except tk.TclError:
            pass
        return tk_dpi

    def _detect_scale(self):
        """Estimate a sensible font scale from the true screen DPI, clamped to a safe range."""
        return round(min(2.5, max(1.3, self._detect_dpi() / 96.0)), 2)

    def _apply_ui_scale(self, event=None):
        """Resize every shared UI font to the current ui_scale (live updates all widgets)."""
        s = self.ui_scale.get()
        for key, (base, bold, mono) in self._font_specs.items():
            self.ui_fonts[key].configure(size=max(8, int(round(base * s))))

    def setup_menu(self):
        # Build the menu bar ourselves: Menubutton (and the native root menu) ignore font sizing
        # under some Linux themes, but a plain Label always honors its font. We pop the menu up
        # manually on click. Uses the shared "menu" font so the UI slider resizes it live.
        menu_font = self.ui_fonts["menu"]
        menubar = tk.Frame(self.root, bg="#e8e8e8", bd=1, relief=tk.RAISED)
        menubar.pack(side=tk.TOP, fill=tk.X)

        file_lbl = tk.Label(menubar, text="File", font=menu_font, bg="#e8e8e8", padx=12, pady=4)
        file_lbl.pack(side=tk.LEFT)

        self.file_menu = tk.Menu(self.root, tearoff=0, font=menu_font)
        self.file_menu.add_command(label="Select Directory...", command=self.browse_workspace)
        self.file_menu.add_command(label="Font Size...", command=self._open_font_settings)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.on_close)

        def _show_file_menu(event):
            self.file_menu.tk_popup(file_lbl.winfo_rootx(),
                                    file_lbl.winfo_rooty() + file_lbl.winfo_height())

        file_lbl.bind("<Button-1>", _show_file_menu)
        # Subtle hover highlight so it reads as a clickable menu
        file_lbl.bind("<Enter>", lambda e: file_lbl.config(bg="#d2d2d2"))
        file_lbl.bind("<Leave>", lambda e: file_lbl.config(bg="#e8e8e8"))

        # View menu: switch between the single-inlet view and the folder-combined view.
        view_lbl = tk.Label(menubar, text="View", font=menu_font, bg="#e8e8e8", padx=12, pady=4)
        view_lbl.pack(side=tk.LEFT)

        self.view_menu = tk.Menu(self.root, tearoff=0, font=menu_font)
        self.view_menu.add_command(label="Pit Hydrograph", command=self.show_pit_hydrograph)
        self.view_menu.add_command(label="Combined Hydrograph", command=self.show_combined_hydrograph)

        def _show_view_menu(event):
            self.view_menu.tk_popup(view_lbl.winfo_rootx(),
                                    view_lbl.winfo_rooty() + view_lbl.winfo_height())

        view_lbl.bind("<Button-1>", _show_view_menu)
        view_lbl.bind("<Enter>", lambda e: view_lbl.config(bg="#d2d2d2"))
        view_lbl.bind("<Leave>", lambda e: view_lbl.config(bg="#e8e8e8"))

    def on_close(self):
        """Cleanly shut down so the process actually exits (closes Matplotlib figures)."""
        self._save_config()
        plt.close('all')
        self.root.quit()
        self.root.destroy()

    def _open_font_settings(self):
        """Popup with the UI and plot font-size sliders (moved here out of the sidebar)."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Font Size")
        dlg.transient(self.root)
        dlg.grab_set()
        pad = tk.Frame(dlg, padx=14, pady=12)
        pad.pack(fill=tk.BOTH, expand=True)

        tk.Label(pad, text="UI Font Size:", font=self.ui_fonts["header"]).pack(anchor=tk.W)
        tk.Scale(
            pad, from_=0.8, to=3.0, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self.ui_scale, length=260, showvalue=True,
            command=lambda v: self._apply_ui_scale(),
        ).pack(fill=tk.X, pady=(0, 10))

        tk.Label(pad, text="Plot Font Size:", font=self.ui_fonts["header"]).pack(anchor=tk.W)
        plot_scale = tk.Scale(
            pad, from_=1.0, to=4.0, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self.font_scale, length=260, showvalue=True,
        )
        plot_scale.pack(fill=tk.X, pady=(0, 12))
        # Re-render the (heavier) plot only on release so dragging stays smooth.
        plot_scale.bind('<ButtonRelease-1>', self.rescale_fonts)

        def close():
            self._save_config()
            dlg.destroy()

        tk.Button(pad, text="Close", font=self.ui_fonts["button"], command=close).pack(anchor=tk.E)
        dlg.protocol("WM_DELETE_WINDOW", close)

        # Widen so the window-manager title ("Font Size") isn't truncated, and lock the size.
        dlg.update_idletasks()
        w = max(dlg.winfo_reqwidth(), 360)
        dlg.geometry(f"{w}x{dlg.winfo_reqheight()}")
        dlg.resizable(False, False)

    def setup_sidebar_controls(self):
        """Creates the directory lookup bar and file tracking selection menu listbox."""
        sidebar = tk.Frame(self.paned, width=int(300 * self.ui_scale.get()), bg="#f5f5f5", padx=10, pady=10)
        sidebar.pack_propagate(False)
        # minsize keeps the controls usable; the user can still drag the sash wider/narrower.
        self.paned.add(sidebar, minsize=int(180 * self.ui_scale.get()), stretch="never")

        # Directory Selection Layout
        dir_lbl = tk.Label(sidebar, text="Target Data Directory:", font=self.ui_fonts["header"], bg="#f5f5f5")
        dir_lbl.pack(anchor=tk.W, pady=(0, 2))

        dir_entry = tk.Entry(sidebar, textvariable=self.selected_dir, width=32, font=self.ui_fonts["body"])
        dir_entry.pack(fill=tk.X, pady=(0, 5))

        btn_browse = tk.Button(sidebar, text="📁 Browse Folder...", command=self.browse_workspace, bg="#e1e1e1", font=self.ui_fonts["button"])
        btn_browse.pack(fill=tk.X, pady=(0, 15))

        # Hydrograph File Selection Listbox Layout
        list_lbl = tk.Label(sidebar, text="Detected Hydrographs:", font=self.ui_fonts["header"], bg="#f5f5f5")
        list_lbl.pack(anchor=tk.W, pady=(0, 2))

        list_frame = tk.Frame(sidebar)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scroll_y = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=scroll_y.set, selectmode=tk.SINGLE, font=self.ui_fonts["mono"])

        scroll_y.config(command=self.file_listbox.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind index clicks directly to data processing functions
        self.file_listbox.bind('<<ListboxSelect>>', self.process_selected_file)

        btn_reload = tk.Button(sidebar, text="🔄 Refresh Workspace", command=self.refresh_workspace_files, bg="#e1e1e1", font=self.ui_fonts["button"])
        btn_reload.pack(fill=tk.X, pady=(10, 0))
        # Font-size sliders now live in File -> Font Size...

    def rescale_fonts(self, event=None):
        """Re-draws whichever view is showing with the newly chosen font scale."""
        self._save_config()
        if self.current_view == "combined":
            self.show_combined_hydrograph()
        elif self.current_df is not None:
            self.generate_hydraulic_plots(self.current_df, self.current_title)

    def setup_graph_display_panel(self):
        """Initialises the primary Matplotlib drawing canvas frame components."""
        self.canvas_frame = tk.Frame(self.paned, bg="#ffffff")
        self.paned.add(self.canvas_frame, minsize=300, stretch="always")

        # Standby message when directory loads without a file actively highlighted
        self.standby_lbl = tk.Label(
            self.canvas_frame,
            text="Please select a hydrograph CSV from the sidebar\nto compile the hydraulic subplots.",
            font=self.ui_fonts["standby"], bg="#ffffff", fg="#555555"
        )
        self.standby_lbl.pack(expand=True)

    def browse_workspace(self):
        """Opens our own directory chooser (the Tk file dialog can't be font-scaled)."""
        chosen_folder = self._ask_directory(self.selected_dir.get())
        if chosen_folder:
            self.selected_dir.set(chosen_folder)
            self.refresh_workspace_files()

    def _ask_directory(self, initialdir):
        """A simple, fully font-scaled directory picker (uses the shared self.ui_fonts)."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Select Directory")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry(f"{int(560 * self.ui_scale.get())}x{int(440 * self.ui_scale.get())}")

        result = {"path": None}
        cur = tk.StringVar(value=os.path.abspath(initialdir if os.path.isdir(initialdir) else os.getcwd()))

        top = tk.Frame(dlg, padx=8, pady=8)
        top.pack(fill=tk.X)
        tk.Label(top, text="Folder:", font=self.ui_fonts["header"]).pack(side=tk.LEFT)
        entry = tk.Entry(top, textvariable=cur, font=self.ui_fonts["body"])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        mid = tk.Frame(dlg, padx=8)
        mid.pack(fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(mid, orient=tk.VERTICAL)
        lb = tk.Listbox(mid, font=self.ui_fonts["mono"], yscrollcommand=sb.set, activestyle="none")
        sb.config(command=lb.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def populate(path):
            lb.delete(0, tk.END)
            lb.insert(tk.END, "..")
            try:
                for name in sorted(os.listdir(path)):
                    if os.path.isdir(os.path.join(path, name)):
                        lb.insert(tk.END, name)
            except OSError:
                pass

        def go(path):
            path = os.path.abspath(path)
            if os.path.isdir(path):
                cur.set(path)
                populate(path)

        lb.bind("<Double-Button-1>",
                lambda e: (lb.curselection() and go(os.path.join(cur.get(), lb.get(lb.curselection())))))
        entry.bind("<Return>", lambda e: go(cur.get()))

        bot = tk.Frame(dlg, padx=8, pady=8)
        bot.pack(fill=tk.X)

        def ok():
            if os.path.isdir(cur.get()):
                result["path"] = cur.get()
            dlg.destroy()

        tk.Button(bot, text="Select", font=self.ui_fonts["button"], command=ok).pack(side=tk.RIGHT)
        tk.Button(bot, text="Cancel", font=self.ui_fonts["button"], command=dlg.destroy).pack(side=tk.RIGHT, padx=(0, 6))

        populate(cur.get())
        dlg.wait_window()
        return result["path"]

    def refresh_workspace_files(self):
        """Scans the directory for CSV logs matching your output formats."""
        self.file_listbox.delete(0, tk.END)
        path = self.selected_dir.get()

        if not os.path.exists(path):
            return

        self.csv_files = [f for f in os.listdir(path) if f.lower().endswith('.csv')]

        for file in sorted(self.csv_files):
            self.file_listbox.insert(tk.END, file)

    def process_selected_file(self, event):
        """Validates structure schemas and coordinates plotting updates."""
        selection = self.file_listbox.curselection()
        if not selection:
            return

        filename = self.file_listbox.get(selection)
        full_filepath = os.path.join(self.selected_dir.get(), filename)

        try:
            df = pd.read_csv(full_filepath)

            # Strict verification step checking for data log schema columns
            required_headers = [
                "Time_s", "Depth_m", "Approach_Q_cms",
                "Captured_Q_cms", "Bypass_Q_cms",
                "Cum_Captured_m3", "Cum_Bypassed_m3"
            ]
            missing_cols = [col for col in required_headers if col not in df.columns]

            if missing_cols:
                messagebox.showerror("Header Format Error", f"CSV lacks required headers:\n{missing_cols}")
                return

            # Keep the selection so the View menu / font slider can re-render it.
            self.current_df = df
            self.current_title = filename
            self.show_pit_hydrograph()

        except Exception as e:
            messagebox.showerror("Parsing Failure", f"Failed to correctly load hydrograph:\n{str(e)}")

    def _clear_standby(self):
        """Remove the placeholder message once something is plotted."""
        if self.standby_lbl:
            self.standby_lbl.pack_forget()
            self.standby_lbl = None

    def show_pit_hydrograph(self):
        """View menu: the single-inlet 4-panel diagnostic plots."""
        if self.current_df is None:
            messagebox.showinfo("No Data", "Select a hydrograph CSV from the sidebar first.")
            return
        self._clear_standby()
        self.generate_hydraulic_plots(self.current_df, self.current_title)

    def show_combined_hydrograph(self):
        """View menu: folder totals — sum Captured/Bypass across every CSV in the
        directory and plot instantaneous flows (L/s) with cumulative volumes (m³)."""
        path = self.selected_dir.get()
        if not os.path.exists(path):
            messagebox.showinfo("No Directory", "Select a valid data directory first.")
            return

        csvs = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.csv')]
        if not csvs:
            messagebox.showinfo("No CSVs", "No CSV files found in the selected directory.")
            return

        # Sum each file's Captured/Bypass onto a common Time_s axis (pure helper).
        combined, skipped = combine_hydrographs(csvs)
        if combined.empty:
            messagebox.showinfo("No Valid Files",
                                "No CSVs with the required hydrograph columns were found.")
            return
        if skipped:
            messagebox.showwarning("Skipped Files",
                                   "Ignored CSVs without hydrograph columns:\n"
                                   + "\n".join(skipped))

        n_files = len(csvs) - len(skipped)
        time_s = combined["Time_s"]
        q_cap_lps = combined["Captured_total_cms"] * 1000.0
        q_byp_lps = combined["Bypass_total_cms"] * 1000.0
        q_comb_lps = combined["Combined_total_cms"] * 1000.0
        cap_cum_m3 = combined["Captured_cum_m3"]
        byp_cum_m3 = combined["Bypass_cum_m3"]
        comb_cum_m3 = combined["Combined_cum_m3"]

        self._clear_standby()
        self.current_view = "combined"

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()

        s = self.font_scale.get()
        suptitle_fs = 10 * s
        title_fs = 10 * s
        label_fs = 9 * s
        legend_fs = 8 * s
        tick_fs = 8 * s

        fig, ax_flow = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), constrained_layout=True)
        ax_vol = ax_flow.twinx()

        fig.suptitle(f"Combined Hydrograph — {n_files} inlet(s) — "
                     f"Max Combined = {q_comb_lps.max():.2f} L/s",
                     fontsize=suptitle_fs, fontweight="bold")

        # Instantaneous flows on the left axis.
        p1, = ax_flow.plot(time_s, q_cap_lps, color="#2e7d32", lw=1.6, label="Captured (L/s)")
        p2, = ax_flow.plot(time_s, q_byp_lps, color="#d32f2f", lw=1.2, linestyle="-.", label="Bypassed (L/s)")
        p3, = ax_flow.plot(time_s, q_comb_lps, color="#1e88e5", lw=1.8, linestyle="--", label="Combined (L/s)")
        ax_flow.set_title("Folder Totals Across All Inlets", fontsize=title_fs, pad=6, loc='left')
        ax_flow.set_xlabel("Elapsed Time (s)", fontsize=label_fs)
        ax_flow.set_ylabel("Flow (L/s)", fontsize=label_fs)
        ax_flow.grid(True, linestyle="--", alpha=0.5)
        ax_flow.tick_params(axis='both', which='major', labelsize=tick_fs)

        # Cumulative volumes on the right axis.
        q1, = ax_vol.plot(time_s, cap_cum_m3, color="#2e7d32", lw=1.0, linestyle=':', label="Captured cum (m³)")
        q2, = ax_vol.plot(time_s, byp_cum_m3, color="#d32f2f", lw=1.0, linestyle=':', label="Bypassed cum (m³)")
        q3, = ax_vol.plot(time_s, comb_cum_m3, color="#1e88e5", lw=1.2, linestyle='-.', label="Combined cum (m³)")
        ax_vol.set_ylabel("Cumulative Volume (m³)", fontsize=label_fs)
        ax_vol.tick_params(axis='y', labelsize=tick_fs)

        lines = [p1, p2, p3, q1, q2, q3]
        ax_flow.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=legend_fs)

        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_hydraulic_plots(self, df, title_label):
        """Generates the four diagnostic subplots into the Tkinter layout canvas using constrained_layout."""
        # Remember what is on screen so the font slider can re-render it
        self.current_df = df
        self.current_title = title_label
        self.current_view = "pit"

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()

        # Use constrained_layout for automatic spacing
        fig, (ax_top, ax_mid, ax_bot, ax_hys) = plt.subplots(nrows=4, ncols=1, figsize=(8, 10), constrained_layout=True)

        # Base font sizes, scaled by the user/DPI-driven font scale so they stay legible on any screen
        s = self.font_scale.get()
        suptitle_fs = 10 * s
        title_fs = 10 * s
        label_fs = 9 * s
        legend_fs = 8 * s
        tick_fs = 8 * s

        # Strip the .csv so the (now larger) title isn't clipped; let constrained_layout place it.
        display_title = title_label[:-4] if title_label.lower().endswith(".csv") else title_label
        fig.suptitle(f"Inlet Hydraulic Analysis — {display_title}", fontsize=suptitle_fs, fontweight='bold')

        time_s = df["Time_s"]
        depth_m = df["Depth_m"]
        q_app_lps = df["Approach_Q_cms"] * 1000.0
        q_cap_lps = df["Captured_Q_cms"] * 1000.0
        q_byp_lps = df["Bypass_Q_cms"] * 1000.0

        # Formatting helper for ticks
        for ax in (ax_top, ax_mid, ax_bot, ax_hys):
            ax.tick_params(axis='both', which='major', labelsize=tick_fs)

        # 1. TOP PLOT: Approach Flow vs Captured Flow
        ax_top.plot(q_app_lps, q_cap_lps, color="#673ab7", lw=1.4, marker='o', markersize=3, label="Operating Performance")
        ax_top.set_title("Approach Flow vs Captured Flow", fontsize=title_fs, pad=6, loc='left')
        ax_top.set_xlabel("Approach Discharge Q_approach (L/s)", fontsize=label_fs)
        ax_top.set_ylabel("Captured Discharge Q_captured (L/s)", fontsize=label_fs)
        ax_top.grid(True, linestyle="--", alpha=0.5)
        max_flow_scale = max(q_app_lps.max(), 10.0)
        ax_top.plot([0, max_flow_scale], [0, max_flow_scale], color="gray", linestyle=":", label="100% Efficiency")
        ax_top.legend(loc="upper left", fontsize=legend_fs)

        # 2. MIDDLE PLOT: Accumulated Volume over Time
        time_deltas = time_s.diff().fillna(0.0)
        cum_approach_m3 = (df["Approach_Q_cms"] * time_deltas).cumsum()

        ax_mid.plot(time_s, cum_approach_m3, color="#0288d1", lw=1.6, label="Accumulated Approach Volume")
        ax_mid.plot(time_s, df["Cum_Captured_m3"], color="#2e7d32", lw=1.6, linestyle="--", label="Accumulated Captured Volume")
        ax_mid.fill_between(time_s, df["Cum_Captured_m3"], cum_approach_m3, color="#d32f2f", alpha=0.12, label="Bypassed Spill")
        ax_mid.set_title("Accumulated Inflow & Captured Volumes (Time)", fontsize=title_fs, pad=6, loc='left')
        ax_mid.set_xlabel("Elapsed Time (s)", fontsize=label_fs)
        ax_mid.set_ylabel("Volume (m³)", fontsize=label_fs)
        ax_mid.grid(True, linestyle="--", alpha=0.5)
        ax_mid.legend(loc="upper left", fontsize=legend_fs)

        # 3. LOWER PLOT: Twin Vertical Axis (Flows & Depth vs Time)
        ax_bot.plot(time_s, q_cap_lps, color="#2e7d32", lw=1.6, label="Captured Flow")
        ax_bot.plot(time_s, q_byp_lps, color="#d32f2f", lw=1.2, linestyle="-.", label="Bypass Flow")
        ax_bot.set_title("Time vs Flows & Depth", fontsize=title_fs, pad=6, loc='left')
        ax_bot.set_xlabel("Elapsed Time (s)", fontsize=label_fs)
        ax_bot.set_ylabel("Flow (L/s)", fontsize=label_fs)
        ax_bot.grid(True, linestyle="--", alpha=0.5)
        ax_bot_twin = ax_bot.twinx()
        ax_bot_twin.plot(time_s, depth_m, color="#ef6c00", lw=2.0, linestyle=":", label="Local Depth")
        ax_bot_twin.set_ylabel("Depth (m)", fontsize=label_fs, color="#ef6c00")
        ax_bot_twin.tick_params(axis='y', labelcolor="#ef6c00", labelsize=tick_fs)
        ax_bot_twin.grid(False)
        # combine legends cleanly
        lines_left, labels_left = ax_bot.get_legend_handles_labels()
        lines_right, labels_right = ax_bot_twin.get_legend_handles_labels()
        ax_bot.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left", fontsize=legend_fs)

        # 4. HYSTERESIS PLOT: Depth vs Approach Discharge (full time)
        ax_hys.plot(q_app_lps.values, depth_m.values, color="#666666", lw=0.9, alpha=0.7, zorder=1)
        sc = ax_hys.scatter(q_app_lps.values, depth_m.values, c=time_s.values, cmap='viridis', s=16, zorder=2)
        ax_hys.set_title("Depth vs Approach Discharge — Hysteresis (full series)", fontsize=title_fs, pad=6, loc='left')
        ax_hys.set_xlabel("Approach Discharge Q_approach (L/s)", fontsize=label_fs)
        ax_hys.set_ylabel("Depth (m)", fontsize=label_fs)
        ax_hys.grid(True, linestyle="--", alpha=0.4)
        # Start and end markers
        ax_hys.scatter([q_app_lps.values[0]], [depth_m.values[0]], color='green', s=36, label='Start', zorder=3)
        ax_hys.scatter([q_app_lps.values[-1]], [depth_m.values[-1]], color='red', s=36, label='End', zorder=3)
        ax_hys.legend(loc='upper left', fontsize=legend_fs)

        # Colorbar positioned using constrained_layout friendly parameters
        cbar = fig.colorbar(sc, ax=[ax_top, ax_mid, ax_bot, ax_hys], orientation='vertical', pad=0.02, fraction=0.035)
        cbar.set_label('Time (s)', fontsize=label_fs)
        cbar.ax.tick_params(labelsize=tick_fs)

        # Draw into Tk canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    """Console entry point: launch the hydrograph viewer GUI."""
    root = tk.Tk()
    HydrographViewerApp(root)
    # Force an initial paint so the split layout gets real frame coordinates.
    root.update_idletasks()
    root.update()
    root.mainloop()


if __name__ == "__main__":
    main()
