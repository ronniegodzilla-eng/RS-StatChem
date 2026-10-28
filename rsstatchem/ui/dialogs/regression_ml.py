"""Analyze > Regression > Partial Least Squares and Kernel Ridge, and Analyze > Generalized
Linear Models > Multiple Adaptive Regression Splines (SPSS PLS, KRR and STATS EARTH dialogs).
Core code: :mod:`rsstatchem.core.stats.pls_regression`, :mod:`~rsstatchem.core.stats.mars`,
:mod:`~rsstatchem.core.stats.kernel_ridge`."""
from __future__ import annotations

from rsstatchem.core.stats import kernel_ridge, mars, pls_regression
from rsstatchem.core.stats.ml_common import measure_of
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.ml_tabs import OptionsDialog, TabbedPickerDialog, parse_list


class PartialLeastSquaresDialog(TabbedPickerDialog):
    DEFAULTS = {"n_factors": 5, "plot_vip": True, "plot_scores": False, "plot_weights": False,
                "save_pred": False, "save_resid": False, "save_dist": False,
                "save_scores": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Partial Least Squares Regression", parent)
        self.add_target("deps", "Dependent Variables:")
        self.add_target("indep", "Independent Variables:")
        self.add_target("id", "Case Identifier Variable:", single=True, required=False)
        self.make_tabs([("Model", self._model), ("Options", self._options)])
        self.finish()

    def _model(self, f):
        g = f.group("Latent Factors")
        g.number("n_factors", "Maximum number of latent factors:", 1, 100)
        f.label("Independent variables with a nominal or ordinal measurement level enter as "
                "factors (indicator columns); scale variables enter as covariates. X and Y are "
                "centred and standardized.")

    def _options(self, f):
        g = f.group("Save to New Variables")
        g.check("save_pred", "Predicted values")
        g.check("save_resid", "Residuals")
        g.check("save_dist", "Distance to latent factor model (DModX)")
        g.check("save_scores", "Latent factor scores")
        g = f.group("Plots")
        g.check("plot_vip", "Cumulative variable importance (VIP) chart")
        g.check("plot_scores", "Latent factor scores (factor 1 by factor 2)")
        g.check("plot_weights", "Latent factor weights")

    def validate(self):
        if set(self.selected("deps")) & set(self.selected("indep")):
            return "A variable cannot be both dependent and independent."
        return None

    def compute(self):
        o = self.opts
        indep = self.selected("indep")
        factors = [v for v in indep if measure_of(self.ds, v) != "scale"]
        covs = [v for v in indep if v not in factors]
        plots = tuple(k for k, on in (("vip", o["plot_vip"]), ("scores", o["plot_scores"]),
                                      ("weights", o["plot_weights"])) if on)
        save = tuple(k for k, on in (("predicted", o["save_pred"]),
                                     ("residuals", o["save_resid"]),
                                     ("distance", o["save_dist"]),
                                     ("scores", o["save_scores"])) if on)
        return pls_regression.pls_procedure(self.ds, self.selected("deps"), factors, covs,
                                            int(o["n_factors"]), self.selected_one("id"),
                                            plots, save)


class MarsDialog(VariablePickerDialog):
    DEFAULTS = {"degree": 1, "penalty_auto": True, "penalty": 2.0, "thresh": 0.001,
                "nk_auto": True, "nk": 21, "nprune_auto": True, "nprune": 10,
                "minspan": 0, "endspan": 0, "family": "gaussian", "plot_selection": True,
                "plot_residuals": True, "plot_partial": True, "save_pred": False,
                "save_resid": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Multiple Adaptive Regression Splines", parent)
        self.add_target("dep", "Dependent Variable:", accepts=is_numeric, single=True)
        self.add_target("pred", "Predictors:")
        self.add_option_button("Model...", self._model)
        self.add_option_button("Plots...", self._plots)
        self.add_option_button("Save...", self._save)
        self.finish()

    def _model(self):
        dlg = OptionsDialog("MARS: Model", self.opts, self)
        g = dlg.group("Model Type")
        g.radio("family", [("gaussian", "Linear (scale dependent)"),
                           ("binomial", "Logistic (binary dependent, GLM refit)")])
        g.number("degree", "Maximum degree of interaction:", 1, 10)
        g = dlg.group("Pruning and Growth")
        g.check("penalty_auto", "Automatic GCV penalty (2 additive, 3 with interactions)")
        g.number("penalty", "Penalty per knot:", 0, 100, 2)
        g.number("thresh", "Forward stepping threshold (RSq change):", 0, 1, 5)
        g.check("nk_auto", "Automatic maximum number of terms (nk)")
        g.number("nk", "Maximum terms in forward pass:", 1, 1000)
        g.check("nprune_auto", "Select the subset by GCV (no limit)")
        g.number("nprune", "Maximum terms after pruning:", 1, 1000)
        g.number("minspan", "Minspan (0 = automatic):", 0, 1e6)
        g.number("endspan", "Endspan (0 = automatic):", 0, 1e6)
        dlg.run(self.opts)

    def _plots(self):
        dlg = OptionsDialog("MARS: Plots", self.opts, self, columns=1)
        g = dlg.group("Plots")
        g.check("plot_selection", "Model selection (GRSq and RSq by number of terms)")
        g.check("plot_residuals", "Residuals vs fitted")
        g.check("plot_partial", "Partial dependence (plotmo)")
        dlg.run(self.opts)

    def _save(self):
        dlg = OptionsDialog("MARS: Save", self.opts, self, columns=1)
        g = dlg.group("Save")
        g.check("save_pred", "Predicted values")
        g.check("save_resid", "Residuals")
        dlg.run(self.opts)

    def compute(self):
        o = self.opts
        plots = tuple(k for k, on in (("selection", o["plot_selection"]),
                                      ("residuals", o["plot_residuals"]),
                                      ("partial", o["plot_partial"])) if on)
        save = tuple(k for k, on in (("predicted", o["save_pred"]),
                                     ("residuals", o["save_resid"])) if on)
        return mars.mars_procedure(
            self.ds, self.selected_one("dep"), self.selected("pred"), int(o["degree"]),
            None if o["nk_auto"] else int(o["nk"]), o["thresh"],
            None if o["penalty_auto"] else o["penalty"], int(o["minspan"]) or None,
            int(o["endspan"]) or None, None if o["nprune_auto"] else int(o["nprune"]),
            o["family"], plots, save)


class KernelRidgeDialog(VariablePickerDialog):
    DEFAULTS = {"kernels": ["rbf"], "alphas": "1", "gammas": "auto", "degrees": "3",
                "coef0s": "1", "folds": 5, "standardize": False, "plot_obs": True,
                "plot_resid": True, "save_pred": False, "save_resid": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Kernel Ridge Regression", parent)
        self.add_target("deps", "Dependent Variables:", accepts=is_numeric)
        self.add_target("pred", "Predictors:")
        self.add_option_button("Kernel...", self._kernel)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Save...", self._save)
        self.finish()

    def _kernel(self):
        dlg = OptionsDialog("Kernel Ridge: Kernel", self.opts, self)
        dlg.checklist("Kernels (several = tuned by cross-validation)", "kernels",
                      list(kernel_ridge.KERNELS.items()))
        g = dlg.group("Parameters (several values separated by spaces)")
        g.text("alphas", "Alpha (regularization):", "1 0.1 0.01")
        g.text("gammas", "Gamma:", "auto = 1 / number of predictors")
        g.text("degrees", "Degree (polynomial):", "3")
        g.text("coef0s", "Coef0 (polynomial, sigmoid):", "1")
        dlg.run(self.opts)

    def _options(self):
        dlg = OptionsDialog("Kernel Ridge: Options", self.opts, self, columns=1)
        g = dlg.group("Cross-validation")
        g.number("folds", "Number of folds:", 2, 50)
        g = dlg.group("Predictors")
        g.check("standardize", "Standardize predictors")
        g = dlg.group("Plots")
        g.check("plot_obs", "Observed by predicted")
        g.check("plot_resid", "Residual by predicted")
        dlg.run(self.opts)

    def _save(self):
        dlg = OptionsDialog("Kernel Ridge: Save", self.opts, self, columns=1)
        g = dlg.group("Save")
        g.check("save_pred", "Predicted values")
        g.check("save_resid", "Residuals")
        dlg.run(self.opts)

    def validate(self):
        o = self.opts
        if not o["kernels"]:
            return "Select at least one kernel."
        try:
            if not parse_list(o["alphas"]):
                return "Specify at least one alpha value."
            parse_list(o["gammas"], allow_none=True)
            parse_list(o["degrees"])
            parse_list(o["coef0s"])
        except ValueError:
            return "Parameter values must be numbers separated by spaces."
        return None

    def compute(self):
        o = self.opts
        plots = tuple(k for k, on in (("observed", o["plot_obs"]),
                                      ("residual", o["plot_resid"])) if on)
        save = tuple(k for k, on in (("predicted", o["save_pred"]),
                                     ("residuals", o["save_resid"])) if on)
        return kernel_ridge.krr_procedure(
            self.ds, self.selected("deps"), self.selected("pred"), list(o["kernels"]),
            parse_list(o["alphas"]), parse_list(o["gammas"], allow_none=True) or [None],
            parse_list(o["degrees"]) or [3], parse_list(o["coef0s"]) or [1.0],
            int(o["folds"]), o["standardize"], plots, save)


MENUS = {"Analyze/Regression": [("Partial Least Squares...", PartialLeastSquaresDialog),
                                ("Kernel Ridge...", KernelRidgeDialog)],
         "Analyze/Generalized Linear Models": [("Multiple Adaptive Regression Splines...",
                                                MarsDialog)]}


def selftest() -> None:
    pls_regression.selftest()
    mars.selftest()
    kernel_ridge.selftest()
