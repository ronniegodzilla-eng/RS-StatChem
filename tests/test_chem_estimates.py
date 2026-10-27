"""Nannoolal / Rarey methods, Marrero-Gani third-order groups, the vapour-pressure estimate and
the CDK module-jar class path (Chemical Drawing -> Chemical Properties)."""
from __future__ import annotations

import math

import pytest
from rdkit import Chem

from rsstatchem.core.chem_draw import cdk, nannoolal as N
from rsstatchem.core.chem_draw.properties import benchmarks, chemical_properties, marrero_gani


def mol(smiles):
    return Chem.MolFromSmiles(smiles)


# --- Nannoolal: group assignment and the published worked examples -------------------------------

# Y. Nannoolal, MSc thesis (UKZN 2004), Appendix C: groups, frequencies and Tb with the thesis'
# own parameters (Appendix D; constants of Table C: Tb = sum / (n^0.6587 + 1.6902) + 84.3359).
MSC_GROUPS = {1: 177.4949, 2: 251.7607, 3: 158.0649, 4: 239.7475, 6: 249.9094, 7: 267.1072,
              15: 235.7455, 16: 315.5639, 21: 53.2649, 34: 390.7067, 42: 223.4973,
              45: 636.9917, 51: 619.5643, 120: -248.0734, 121: -20.3435, 123: -172.7072,
              128: -4.1349, 133: 121.3234, 134: 40.8249}     # 134 = the thesis' C=C-C=O (118)
MSC_INTER = {("OH", "OH"): 292.2832, ("OH", "NH"): 287.5930}
MSC = {"groups": MSC_GROUPS, "interactions": MSC_INTER, "constants": (0.6587, 1.6902, 84.3359)}


@pytest.mark.parametrize("smiles,groups,corrections,tb_msc,tb_2004", [
    # Table C-1, 3,3,4,4-tetramethylhexane (steric correction C3C-CC3)
    ("CCC(C)(C)C(C)(C)CC", {1: 6, 4: 2, 6: 2}, {133: 1}, 431.0, 429.5),
    # Table C-2, di-isopropanolamine (interactions OH-OH 1/9, OH-NH 2/9)
    ("CC(O)CNCC(C)O", {1: 2, 7: 4, 34: 2, 42: 1}, {}, 509.2, 509.3),
    # Table C-3, perfluoro-2-propanone
    ("FC(F)(F)C(=O)C(F)(F)F", {7: 2, 21: 6, 51: 1}, {120: 1, 121: 2, 123: 1}, 246.3, 246.3),
    # Table C-4, methyl m-toluate (meta pair, C=C-C=O)
    ("COC(=O)c1cccc(C)c1", {2: 1, 3: 1, 15: 4, 16: 2, 45: 1}, {128: 1, 134: 1}, 490.5, 490.7),
])
def test_nannoolal_boiling_point_examples(smiles, groups, corrections, tb_msc, tb_2004):
    res = N.fragment(mol(smiles), tb_set=True)
    assert res.ok and res.groups == groups and res.corrections == corrections
    assert N.estimate(res, "tb", params=MSC)[0] == pytest.approx(tb_msc, abs=0.06)
    # 2004 publication: same groups, published values and constants (its Table 17a-d)
    assert N.estimate(res, "tb")[0] == pytest.approx(tb_2004, abs=0.06)


def test_nannoolal_interaction_frequencies_follow_the_thesis():
    res = N.fragment(mol("CC(O)CNCC(C)O"), tb_set=True)
    freq = res.interaction_counts()
    assert freq[("OH", "OH")] == pytest.approx(1) and freq[("NH", "OH")] == pytest.approx(2)


def test_nannoolal_critical_properties_phd_examples():
    """PhD thesis (UKZN 2006), Appendix D, Tables D-1 to D-3."""
    tmb = N.nannoolal(mol("CC(C)(C)C(C)(C)C"), tb=379.6)
    assert tmb["frag"].groups == {1: 6, 6: 2} and tmb["frag"].corrections == {133: 1}
    assert tmb["tc"] == pytest.approx(566.8, abs=0.06)
    deg = N.nannoolal(mol("COCCOCCO"))            # diethylene glycol monomethyl ether
    assert deg["frag"].groups == {2: 1, 7: 4, 35: 1, 38: 2}
    assert deg["pc"] * 100 == pytest.approx(3689.9, abs=0.5)          # kPa
    tcs = N.nannoolal(mol("Cl[SiH](Cl)Cl"))        # trichlorosilane: Si(C,H)1 + Si-halogen
    assert tcs["frag"].groups == {27: 3, 215: 1} and tcs["frag"].corrections == {124: 1, 217: 1}
    assert tcs["vc"] == pytest.approx(262.9, abs=0.1)


def test_nannoolal_vapour_pressure_examples():
    # PhD thesis Table D-4: perfluoro-2-propanone, dB and P(210.16 K) with Tb = 245.9 K
    pfp = N.nannoolal(mol("FC(F)(F)C(=O)C(F)(F)F"), tb=245.9)
    assert pfp["db"] == pytest.approx(0.1179392, abs=1e-6)
    assert N.vapour_pressure(210.16, 245.9, pfp["db"]) / 1000 == pytest.approx(14.63, abs=0.01)
    assert pfp["tc"] == pytest.approx(358.0, abs=0.1)
    # examples of Nannoolal et al., Fluid Phase Equilib. 269 (2008), Table 15 (values as
    # reproduced in pychemqt's test suite): dB and P at T with the experimental Tb
    for smiles, tb, db, t, p_kpa in (
            ("CC1=CCC2CC1C2(C)C", 429.0, 0.1078596, 388.15, 31.03),       # alpha-pinene
            ("OCCO", 470.5, 1.1310491, 410.65, 13.05),                    # 1,2-ethanediol
            ("C=CC(=O)O", 413.6, 0.9163297, 344.15, 6.52),                # acrylic acid
            ("CC(=O)OCCO", 458.65, 0.5111422, None, None),                # glycol monoacetate
            ("CCCOC(=O)CCC(=O)OCCC", 500.0, 0.9878298, None, None),       # dipropyl succinate
            ("ClC(Cl)C(F)(F)Cl", 344.25, 0.0447374, 297.46, 17.5093)):    # R122
        r = N.nannoolal(mol(smiles))
        assert r["db"] == pytest.approx(db, abs=1e-6), smiles
        if t:
            assert N.vapour_pressure(t, tb, r["db"]) / 1000 == pytest.approx(p_kpa, rel=2e-4)
    dea = N.nannoolal(mol("OCCNCCO"))                                     # diethanolamine
    assert N.vapour_pressure(401.13, 541.15, dea["db"]) / 1000 == pytest.approx(0.40405, rel=1e-4)
    # o-xylene, Tc / Pc / Vc with Tb = 417.58 K (textbook example quoted by pychemqt)
    xyl = N.nannoolal(mol("Cc1ccccc1C"), tb=417.58)
    assert xyl["frag"].corrections == {127: 1}
    assert (xyl["tc"], xyl["pc"] * 100, xyl["vc"]) == pytest.approx((628.87, 3623.55, 370.57),
                                                                    rel=5e-5)   # rounding, M


def test_nannoolal_refuses_what_it_does_not_cover():
    assert not N.fragment(mol("CC(=O)C(C)=O")).ok                 # 1,2-diketone
    assert "imidazole" in N.fragment(mol("c1c[nH]cn1")).missing
    assert not N.fragment(mol("CN=NC")).ok                        # azo compound
    assert not N.fragment(mol("[Na+].[Cl-]")).ok
    assert N.estimate(N.fragment(mol("CNC(C)=O")), "tc")[0] is None   # -CONH-: no Tc value
    assert N.fragment(mol("O=C(NC1CC2(C1)OCCO2)c1ccc(F)c(Cl)c1"), tb_set=True).ok


def test_solid_correction_walden_rule():
    assert N.solid_correction(298.15, 250.0) == 0.0 and N.solid_correction(298.15, None) == 0.0
    # 56.5 J/(mol K) * 100 K / (ln 10 * R * 298.15 K)
    assert N.solid_correction(298.15, 398.15) == pytest.approx(-0.9898, abs=1e-3)


# --- Marrero-Gani third-order groups -------------------------------------------------------------

@pytest.mark.parametrize("smiles,prop,second,third,expected,groups", [
    # Marrero & Gani 2001 worked examples (as reproduced in pychemqt): N-phenyl-1,4-
    # benzenediamine (aC-NH-aC), pyrene (AROMFUSED[3] x2, [4p] x2), 4-aminobutanol,
    # 1,9-nonanediol (Tm), 3,3'-methylenebis(4-hydroxycoumarin) (Tm, first + third order)
    ("Nc1ccc(Nc2ccccc2)cc1", "tb", True, True, 623.94, {"ac_nh_ac": 1}),
    ("c1cc2ccc3cccc4ccc(c1)c2c34", "tb", True, True, 673.96, {"af3": 2, "af4p": 2}),
    ("NCCCCO", "tb", True, True, 478.42, {"aminoalcohol": 1}),
    ("OCCCCCCCCCO", "tm", True, True, 315.70, {"diol": 1}),
    ("O=c1oc2ccccc2c(O)c1Cc1c(O)c2ccccc2oc1=O", "tm", False, True, 552.34,
     {"af2": 2, "ac_cc_cyc_fused": 2, "ac_o_cyc_fused": 2, "cyc_ch_cyc": 1}),
    # flutemazepam, Tm with all three orders (aC-NHn(cyclic) fused, aC-(C=N)cyc, ...)
    ("CN1C(=O)C(O)N=C(c2ccccc2F)c2cc(Cl)ccc21", "tm", True, True, 419.63,
     {"ac_nh_cyc_fused": 1, "ac_cn_cyc_diff": 1, "ac_cn_cyc_fused": 1, "af2_s2": 1}),
])
def test_marrero_gani_third_order_examples(smiles, prop, second, third, expected, groups):
    res = marrero_gani(mol(smiles), second_order=second, third_order=third)
    assert res.ok and res.third == groups
    assert res.values[prop] == pytest.approx(expected, abs=0.01)


def test_marrero_gani_third_order_ring_patterns():
    third = marrero_gani(mol("c1ccc2ncccc2c1"), third_order=True).third
    assert third == {"af2": 1, "pf2": 1}                           # quinoline
    assert marrero_gani(mol("Cc1ccc2ccccc2c1"), third_order=True).third == {"af2_s2": 1, "af2": 1}
    assert marrero_gani(mol("C1CCC2CCCCC2C1"), third_order=True).third == {"ch_multiring": 2}
    assert marrero_gani(mol("c1ccc(-c2ccccc2)cc1"), third_order=True).third == {"ac_ac": 1}
    assert marrero_gani(mol("OCCOCCO"), third_order=True).third == {"oh_o_oh": 1}
    assert marrero_gani(mol("OCCO"), third_order=True).third == {}           # m > 2 only


# --- benchmark-driven defaults and the vapour-pressure row ---------------------------------------

def test_benchmark_has_the_new_methods_and_picks_defaults():
    b = benchmarks()
    assert {"nan", "nan_tbest", "mg3"} <= set(b["tc"]["methods"])
    assert "nan" in b["bp"]["methods"] and "mg3" in b["pc"]["methods"]
    assert b["logvp"]["methods"]["nan_s"]["n"] > 1000
    assert set(b["logvp"]["subsets"]) >= {"tb_experimental_liquids", "tb_estimated_solids"}
    rows = {p.name: p for p in chemical_properties(mol("CCCCCCC(C)OC(=O)CC"))}
    best_tc = min(("nan", "mg", "mg3", "mg1", "ag", "joback"),
                  key=lambda m: b["tc"]["common"]["methods"][m if m != "nan" else
                                                             "nan_tbest"]["rmse"])
    assert rows["Critical Temp."].alternatives[[a.default for a in
                                                rows["Critical Temp."].alternatives
                                                ].index(True)].key == best_tc
    tc = rows["Critical Temp."]
    if tc.short == "Nannoolal":
        assert "Tb" in tc.detail


def test_vapour_pressure_row_estimate_and_experimental():
    rows = {p.name: p for p in chemical_properties(mol("CCCCCCC(C)OC(=O)CC"))}
    vp = rows["Vapor Pres."]
    assert vp.kind == "estimate" and vp.short == "Nannoolal" and vp.value > 0
    assert "Tb" in vp.detail and "expected" in vp.detail and "Pa at 25" in vp.text()
    # a solid gets the Walden conversion, the experimental value wins when present
    novel = {p.name: p for p in chemical_properties(
        mol("O=C(NC1CC2(C1)OCCO2)c1ccc(F)c(Cl)c1"))}["Vapor Pres."]
    assert novel.value is not None and "Walden" in novel.detail
    tol = {p.name: p for p in chemical_properties(mol("Cc1ccccc1"))}["Vapor Pres."]
    assert tol.kind == "experimental" and math.log10(tol.value) == pytest.approx(3.58, abs=0.1)
    nan_alt = [a for a in tol.alternatives if a.key == "nan"][0]
    assert nan_alt.value == pytest.approx(tol.value, rel=0.5) and not nan_alt.default


# --- CDK class path ------------------------------------------------------------------------------

@pytest.mark.skipif(cdk.cdk_jars() is None, reason="CDK jars not fetched")
def test_cdk_class_path_is_the_helper_plus_module_jars():
    jars = cdk.cdk_jars()
    names = [j.name for j in jars]
    assert names[0] == "rsstatchem-cdk.jar" and "cdk.jar" not in names
    assert {"cdk-core-2.13.jar", "cdk-qsarmolecular-2.13.jar", "beam-core-1.3.10.jar",
            "vecmath-1.5.2.jar"} <= set(names)
    assert sum(j.stat().st_size for j in jars[1:]) < 3e6           # 24 MB bundle before
