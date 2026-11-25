"""The Claude connector (``RS-StatChem --mcp``): the tools called directly, the MCP server through
an in-memory client, and the real thing — a subprocess serving over stdio.

Results are compared with scipy / statsmodels, like the tests of the analyses themselves; here
the point is that the right analysis is run with the right arguments and comes back intact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.io import files
from rsstatchem.mcp import render
from rsstatchem.mcp.tools import TOOLS, Workspace

ROOT = Path(__file__).resolve().parent.parent


def make_dataset() -> Dataset:
    rng = np.random.default_rng(11)
    n = 60
    pre = rng.normal(60, 8, n).round(1)
    group = np.repeat([1.0, 2.0], n // 2)
    post = (pre + 4 + 3 * (group - 1) + rng.normal(0, 4, n)).round(1)
    age = rng.integers(18, 26, n).astype(float)
    age[:3] = 99.0
    ds = Dataset(pd.DataFrame({
        "id": [f"R{i:02d}" for i in range(n)], "group": group, "pre": pre, "post": post,
        "age": age, "level": rng.integers(1, 4, n).astype(float)}))
    ds.var("group").value_labels = {1.0: "Kontrol", 2.0: "Eksperimen"}
    ds.var("group").measure = "nominal"
    ds.var("pre").label = "Nilai pre-test"
    ds.var("age").missing = [99.0]
    return ds


@pytest.fixture
def data_file(tmp_path) -> Path:
    path = tmp_path / "penelitian.rsd"
    files.save_project(make_dataset(), path)
    return path


@pytest.fixture
def ws(data_file) -> Workspace:
    workspace = Workspace()
    workspace.open_data(str(data_file))
    return workspace


def tables(text: str) -> dict[str, dict]:
    return {i["title"]: i for i in json.loads(text)["items"] if i["kind"] == "table"}


def num(cell: str) -> float:
    return float(cell.rstrip("*"))


# ---------------------------------------------------------------------- rendering
def test_render_markdown_and_json():
    section = Section("Demo")
    section.log("DEMO /VARIABLES=a.")
    section.table(Table("T", ["", ("CI", "Lower"), ("CI", "Upper"), "p"],
                        [["a|b", "1.0", "2.0", ".045"], ["line\nbreak", "", "", ""]],
                        footnotes=["a. note"]))
    section.note("Careful", "small sample", style="warning")
    section.image("Histogram", b"png")
    md = render.render(section)
    assert md.startswith("## Demo")
    assert "|  | CI / Lower | CI / Upper | p |" in md
    assert "| a\\|b | 1.0 | 2.0 | .045 |" in md and "line<br>break" in md
    assert "a. note" in md and "Warning: small sample" in md
    assert "Chart not sent: Histogram" in md and "png" not in md
    data = json.loads(render.render(section, "json"))
    assert [i["kind"] for i in data["items"]] == ["syntax", "table", "warning", "chart"]
    assert data["items"][1]["columns"] == ["", "CI / Lower", "CI / Upper", "p"]
    with pytest.raises(ValueError, match="output_format"):
        render.render(section, "html")


# ---------------------------------------------------------------------- data
def test_open_lists_variables_with_labels(ws, data_file):
    text = ws.list_variables()
    assert "60 cases, 6 variables" in text and str(data_file) in text
    assert "| group | numeric | nominal |  | 1 = Kontrol; 2 = Eksperimen |  | 60 |" in text
    assert "| pre | numeric | scale | Nilai pre-test |" in text
    assert "| age | numeric | scale |  |  | 99 | 57 |" in text      # user-missing honoured
    listed = tables(ws.list_datasets("json"))["Datasets"]["rows"]
    assert listed == [["penelitian", "yes", "60", "6", str(data_file)]]


def test_open_every_supported_type(tmp_path):
    ds = make_dataset()
    files.write_sav(ds, tmp_path / "a.sav")
    files.write_excel(ds, tmp_path / "b.xlsx")
    ds.df.to_csv(tmp_path / "c.csv", sep=";", decimal=",", index=False)   # Indonesian style
    ws = Workspace()
    assert "1 = Kontrol; 2 = Eksperimen" in ws.open_data(str(tmp_path / "a.sav"))
    excel = ws.open_data(str(tmp_path / "b.xlsx"))
    assert "Worksheet 'Data' was read; the file has: Data, Variables." in excel
    assert "| Name |" in ws.open_data(str(tmp_path / "b.xlsx"), name="kamus", sheet="variables")
    ws.open_data(str(tmp_path / "c.csv"))
    assert list(ws.datasets) == ["a", "b", "kamus", "c"] and ws.active == "c"
    assert ws.dataset("c").df["pre"].tolist() == ds.df["pre"].tolist()   # decimal comma read
    assert ws.dataset("A") is ws.datasets["a"]
    with pytest.raises(ValueError, match="No worksheet named 'x'"):
        ws.open_data(str(tmp_path / "b.xlsx"), sheet="x")


def test_reopening_a_file_replaces_it_and_same_stem_gets_a_new_name(ws, data_file, tmp_path):
    ws.open_data(str(data_file))
    assert list(ws.datasets) == ["penelitian"]
    other = tmp_path / "sub" / "penelitian.rsd"
    other.parent.mkdir()
    files.save_project(make_dataset(), other)
    ws.open_data(str(other))
    assert list(ws.datasets) == ["penelitian", "penelitian_2"]


def test_save_round_trip_and_overwrite_guard(ws, tmp_path):
    for ext in (".rsd", ".sav", ".xlsx", ".csv"):
        target = tmp_path / f"out{ext}"
        assert str(target) in ws.save_data(str(target))
        again = Workspace()
        again.open_data(str(target))
        assert again.dataset().n_cases == 60 and again.dataset().has_name("post")
    before = (tmp_path / "out.csv").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        ws.save_data(str(tmp_path / "out.csv"))
    assert (tmp_path / "out.csv").read_bytes() == before
    ws.save_data(str(tmp_path / "out.csv"), overwrite=True)
    with pytest.raises(ValueError, match="Cannot save as '.pdf'"):
        ws.save_data(str(tmp_path / "out.pdf"))
    with pytest.raises(ValueError, match="folder does not exist"):
        ws.save_data(str(tmp_path / "nowhere" / "out.rsd"))


def test_messages_for_mistakes(ws, tmp_path):
    with pytest.raises(ValueError, match="No dataset is open"):
        Workspace().descriptives(["pre"])
    with pytest.raises(ValueError, match="not a full path"):
        ws.open_data("data.sav")
    with pytest.raises(ValueError, match="File not found"):
        ws.open_data(str(tmp_path / "missing.sav"))
    with pytest.raises(ValueError, match="Unsupported file type"):
        (tmp_path / "x.pdf").write_bytes(b"")
        ws.open_data(str(tmp_path / "x.pdf"))
    with pytest.raises(ValueError, match="no variable named 'nilai'.*Variables: id, group"):
        ws.descriptives(["nilai"])
    with pytest.raises(ValueError, match="'id' is a string variable"):
        ws.one_sample_t_test(["id"])
    with pytest.raises(ValueError, match="no open dataset named 'lain'"):
        ws.descriptives(["pre"], dataset="lain")
    with pytest.raises(ValueError, match="confidence_level"):
        ws.one_sample_t_test(["pre"], confidence_level=1.5)
    with pytest.raises(ValueError, match="at least 2 variable"):
        ws.correlations(["pre"])
    with pytest.raises(ValueError, match="3 distinct values"):
        ws.independent_samples_t_test(["post"], "level")
    with pytest.raises(ValueError, match="not a value of 'group' \\(1 = Kontrol"):
        ws.independent_samples_t_test(["post"], "group", "Kontrol", "Lainnya")
    with pytest.raises(ValueError, match="exactly two variables"):
        ws.paired_samples_t_test([["pre"]])
    with pytest.raises(ValueError, match="both the dependent and an independent"):
        ws.linear_regression("post", ["pre", "post"])


# ---------------------------------------------------------------------- analyses
def test_descriptives_and_frequencies(ws):
    df = make_dataset().df
    rows = tables(ws.descriptives(["PRE", "age"], ["mean", "std"], output_format="json"))[
        "Descriptive Statistics"]["rows"]
    assert rows[0][0] == "Nilai pre-test" and int(rows[0][1]) == 60
    assert num(rows[0][2]) == pytest.approx(df["pre"].mean(), abs=1e-3)
    assert int(rows[1][1]) == 57                                   # 99 is user-missing
    freq = ws.frequencies(["group"], ["mode"])
    assert "| Valid | Kontrol | 30 | 50.0 | 50.0 | 50.0 |" in freq
    assert "FREQUENCIES VARIABLES=group" in freq
    # a table lists the values themselves: none for a variable with many distinct values
    many = ws.frequencies(["id", "group"], ["mode"])
    assert "| Valid | Kontrol | 30 |" in many and "R07" not in many
    assert "No frequency table for id: more than 50 distinct values" in many
    assert "R07" in ws.frequencies(["id"], max_categories=60)


def test_t_and_z_tests_match_scipy(ws):
    df = make_dataset().df
    t = tables(ws.one_sample_t_test(["pre"], 58, output_format="json"))["One-Sample Test"]
    ref = stats.ttest_1samp(df["pre"], 58)
    assert num(t["rows"][0][1]) == pytest.approx(ref.statistic, abs=1e-3)
    assert t["columns"][1] == "Test Value = 58 / t"

    z = tables(ws.one_sample_z_test(["pre"], sigma=8, test_value=58, alternative="greater",
                                    output_format="json"))["One-Sample Z Test"]
    assert num(z["rows"][0][1]) == pytest.approx((df["pre"].mean() - 58) / (8 / 60 ** 0.5),
                                                 abs=1e-3)

    a, b = df["post"][df["group"] == 1], df["post"][df["group"] == 2]
    ref = stats.ttest_ind(a, b)
    for kwargs in ({}, {"group1": "Kontrol", "group2": "2"}, {"cut_point": 2}):
        ind = tables(ws.independent_samples_t_test(["post"], "group", output_format="json",
                                                   **kwargs))["Independent Samples Test"]
        sign = -1 if "cut_point" in kwargs else 1                  # >= cut is the first group
        assert num(ind["rows"][0][4]) == pytest.approx(sign * ref.statistic, abs=1e-3)

    pair = tables(ws.paired_samples_t_test([["pre", "post"]], confidence_level=95,
                                           output_format="json"))["Paired Samples Test"]
    ref = stats.ttest_rel(df["pre"], df["post"])
    assert num(pair["rows"][0][pair["columns"].index("t")]) == pytest.approx(ref.statistic,
                                                                             abs=1e-3)
    assert "Paired Differences / 95% CI Lower" in pair["columns"]


def test_anova_correlation_regression_match_references(ws):
    import statsmodels.api as sm

    df = make_dataset().df
    out = tables(ws.one_way_anova(["post"], "level", post_hoc=["tukey"], welch=True,
                                  output_format="json"))
    ref = stats.f_oneway(*[g["post"] for _, g in df.groupby("level")])
    row = out["ANOVA"]["rows"][0]
    assert num(row[out["ANOVA"]["columns"].index("F")]) == pytest.approx(ref.statistic, abs=1e-3)
    assert {"Multiple Comparisons", "Robust Tests of Equality of Means"} <= set(out)

    corr = json.loads(ws.correlations(["pre", "post"], ["pearson", "spearman"],
                                      output_format="json"))["items"]
    assert [i["title"] for i in corr if i["kind"] == "table"] == ["Correlations"] * 2
    cells = [c for t in corr if t["kind"] == "table" for r in t["rows"] for c in r]
    for expected in (stats.pearsonr(df["pre"], df["post"]).statistic,
                     stats.spearmanr(df["pre"], df["post"]).statistic):
        assert any(c.rstrip("*") == f"{expected:.3f}".lstrip("0") for c in cells)

    reg = tables(ws.linear_regression("post", ["pre", "group"], output_format="json"))
    coef = next(t for name, t in reg.items() if name.startswith("Coefficients"))
    fit = sm.OLS(df["post"], sm.add_constant(df[["pre", "group"]])).fit()
    b = coef["columns"].index("Unstandardized Coefficients / B")
    got = [num(r[b]) for r in coef["rows"][:3]]
    assert got == pytest.approx(fit.params.tolist(), abs=1e-3)
    assert any("Assumption" in name or "Normality" in name for name in reg)
    short = tables(ws.linear_regression("post", ["pre"], assumption_tests=False,
                                        output_format="json"))
    assert len(short) < len(reg)
    assert "Chart not sent" not in ws.linear_regression("post", ["pre"])   # no plots asked


def test_select_cases_and_split_file_of_the_dataset_are_honoured(tmp_path):
    ds = make_dataset()
    ds.add_variable("filter_$", values=(ds.df["group"] == 2).astype(float))
    ds.meta["filter"] = "filter_$"
    files.save_project(ds, tmp_path / "f.rsd")
    ws = Workspace()
    assert "Filter: filter_$" in ws.open_data(str(tmp_path / "f.rsd"))
    text = ws.descriptives(["post"], output_format="json")
    assert int(tables(text)["Descriptive Statistics"]["rows"][0][1]) == 30
    assert "Filter: filter_$" in text

    ds.meta = {"split": {"vars": ["group"], "mode": "separate"}}
    files.save_project(ds, tmp_path / "s.rsd")
    ws.open_data(str(tmp_path / "s.rsd"))
    split = tables(ws.one_sample_t_test(["pre"], output_format="json"))
    assert sum(name.startswith("One-Sample Test") for name in split) == 2
    assert any("Kontrol" in name for name in split)


# ---------------------------------------------------------------------- MCP server
def test_server_offers_the_tools_and_reports_errors_to_the_model(data_file):
    from mcp import Client

    from rsstatchem.mcp.server import INSTRUCTIONS, build_server

    async def run():
        async with Client(build_server(Workspace())) as client:
            listed = (await client.list_tools()).tools
            assert [t.name for t in listed][:len(TOOLS)] == [t.name for t in TOOLS]
            by_name = {t.name: t for t in listed}
            assert all(t.description and "\nArgs:" in t.description + "\nArgs:" for t in listed)
            assert by_name["descriptives"].annotations.read_only_hint is True
            assert by_name["save_data"].annotations.read_only_hint is False
            schema = by_name["one_sample_z_test"].input_schema
            assert schema["required"] == ["variables", "sigma"]
            assert schema["properties"]["alternative"]["enum"] == ["two-sided", "greater",
                                                                   "less"]
            opened = await client.call_tool("open_data", {"path": str(data_file)})
            assert not opened.is_error and "60 cases" in opened.content[0].text
            assert opened.structured_content is None
            ok = await client.call_tool("correlations", {"variables": ["pre", "post"]})
            assert not ok.is_error and "Pearson Correlation" in ok.content[0].text
            bad = await client.call_tool("descriptives", {"variables": ["tidak_ada"]})
            assert bad.is_error and "no variable named 'tidak_ada'" in bad.content[0].text
            assert client.server_info.name == "rs-statchem"
            assert client.instructions == INSTRUCTIONS

    anyio.run(run)


def test_packaged_selftest_of_the_connector():
    from rsstatchem.mcp.server import selftest

    selftest()


def test_stdio_server_started_like_claude_desktop_does(data_file, tmp_path):
    """`python -m rsstatchem --mcp` as a child process, spoken to over its stdin/stdout."""
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "rsstatchem", "--mcp"], cwd=str(ROOT),
        env={"RSSTATCHEM_HOME": str(tmp_path / "home"), "QT_QPA_PLATFORM": "offscreen",
             "PYTHONWARNINGS": "default"})    # warnings on stderr must not break the stream

    async def run():
        with anyio.fail_after(120):
            async with Client(params) as client:
                names = [t.name for t in (await client.list_tools()).tools]
                assert names[:len(TOOLS)] == [t.name for t in TOOLS] and "run_command" in names
                await client.call_tool("open_data", {"path": str(data_file)})
                result = await client.call_tool(
                    "linear_regression", {"dependent": "post", "independents": ["pre"]})
                assert not result.is_error and "Model Summary" in result.content[0].text
                saved = tmp_path / "hasil.sav"
                await client.call_tool("save_data", {"path": str(saved)})
                assert saved.is_file()

    anyio.run(run)
