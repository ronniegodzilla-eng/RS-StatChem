"""Build the offline structure <-> name database used by Convert Structure to Name.

The data come from PubChem PUG-REST (https://pubchem.ncbi.nlm.nih.gov/rest/pug), whose
compound records are public domain. For the lowest PubChem CIDs (the long-established
compounds) plus a curated list of common names, the script fetches the InChIKey, the IUPAC
name, the PubChem title (the common name) and the SMILES, and writes a compact,
LZMA-compressed TSV sorted by InChIKey::

    rsstatchem/resources/names/name_db.tsv.xz
    # header lines
    INCHIKEY <tab> IUPAC name <tab> common name ("" = same as IUPAC) <tab> SMILES

PubChem asks for at most 5 requests per second (and 400 per minute); the script stays well
below that (``--rate``, default 3/s), backs off when the server says it is busy (HTTP 503 or
a red ``X-Throttling-Control`` header) and caches every answer, so an interrupted run resumes
where it stopped. When the compressed file would exceed ``--max-mb``, the highest CIDs are
dropped (curated names are always kept).

Run from the project root::

    python scripts/build_name_db.py --max-cid 120000 --cache ~/.cache/rsstatchem-names
"""
from __future__ import annotations

import argparse
import json
import lzma
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rsstatchem" / "resources" / "names" / "name_db.tsv.xz"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = "InChIKey,IUPACName,Title,SMILES"
UA = "RS-StatChem name-database builder (education; contact via GitHub)"
MAX_IUPAC = 160            # longer systematic names are rarely useful in a drawing caption
MAX_SMILES = 160

# Common names a chemistry course uses; resolved to CIDs by name (kept whatever their CID).
CURATED = """
water; ammonia; hydrogen peroxide; carbon dioxide; carbon monoxide; ozone; hydrazine;
hydrogen chloride; hydrogen fluoride; hydrogen bromide; hydrogen iodide; hydrogen sulfide;
sulfuric acid; nitric acid; phosphoric acid; hydrochloric acid; perchloric acid; boric acid;
carbonic acid; sodium hydroxide; potassium hydroxide; calcium hydroxide; sodium chloride;
sodium bicarbonate; sodium carbonate; potassium permanganate; potassium dichromate; methane;
ethane; propane; butane; pentane; hexane; heptane; octane; nonane; decane; isobutane;
isopentane; neopentane; isooctane; cyclopropane; cyclobutane; cyclopentane; cyclohexane;
cycloheptane; cyclooctane; cyclohexene; cyclopentene; 1,3-butadiene; isoprene; ethylene;
propylene; 1-butene; 2-butene; acetylene; propyne; benzene; toluene; o-xylene; m-xylene;
p-xylene; ethylbenzene; cumene; styrene; mesitylene; naphthalene; anthracene; phenanthrene;
pyrene; biphenyl; azulene; indene; fluorene; methanol; ethanol; 1-propanol; 2-propanol;
1-butanol; 2-butanol; tert-butanol; isobutanol; 1-pentanol; 1-hexanol; 1-octanol;
cyclohexanol; benzyl alcohol; phenol; ethylene glycol; propylene glycol; glycerol; sorbitol;
mannitol; xylitol; erythritol; catechol; resorcinol; hydroquinone; pyrogallol; cresol;
o-cresol; m-cresol; p-cresol; thymol; carvacrol; menthol; borneol; camphor; geraniol;
linalool; citral; limonene; alpha-pinene; beta-pinene; myrcene; formaldehyde; acetaldehyde;
propionaldehyde; butyraldehyde; benzaldehyde; cinnamaldehyde; vanillin; salicylaldehyde;
furfural; acrolein; glyoxal; glutaraldehyde; acetone; butanone; cyclohexanone; acetophenone;
benzophenone; acetylacetone; methyl vinyl ketone; mesityl oxide; formic acid; acetic acid;
propionic acid; butyric acid; valeric acid; caproic acid; caprylic acid; lauric acid;
myristic acid; palmitic acid; stearic acid; oleic acid; linoleic acid; linolenic acid;
arachidonic acid; benzoic acid; salicylic acid; phthalic acid; terephthalic acid; oxalic
acid; malonic acid; succinic acid; glutaric acid; adipic acid; maleic acid; fumaric acid;
lactic acid; pyruvic acid; citric acid; malic acid; tartaric acid; glycolic acid; acrylic
acid; methacrylic acid; cinnamic acid; gallic acid; ascorbic acid; uric acid; barbituric
acid; trichloroacetic acid; trifluoroacetic acid; chloroacetic acid; methanesulfonic acid;
p-toluenesulfonic acid; acetic anhydride; acetyl chloride; benzoyl chloride; phosgene;
thionyl chloride; oxalyl chloride; methyl acetate; ethyl acetate; butyl acetate; isoamyl
acetate; ethyl formate; methyl formate; ethyl benzoate; methyl benzoate; methyl salicylate;
dimethyl carbonate; diethyl carbonate; ethylene carbonate; propylene carbonate;
gamma-butyrolactone; diethyl malonate; ethyl acetoacetate; diethyl ether; diisopropyl ether;
methyl tert-butyl ether; tetrahydrofuran; 1,4-dioxane; anisole; 1,2-dimethoxyethane;
diglyme; furan; ethylene oxide; propylene oxide; epichlorohydrin; acetonitrile;
benzonitrile; propionitrile; acrylonitrile; malononitrile; hydrogen cyanide; dimethyl
sulfoxide; dimethyl sulfide; sulfolane; carbon disulfide; thiophene; ethanethiol;
N,N-dimethylformamide; N,N-dimethylacetamide; N-methyl-2-pyrrolidone;
hexamethylphosphoramide; formamide; acetamide; benzamide; urea; thiourea; guanidine;
melamine; biuret; dichloromethane; chloroform; carbon tetrachloride; chloromethane;
bromomethane; iodomethane; 1,2-dichloroethane; 1,1,1-trichloroethane; trichloroethylene;
tetrachloroethylene; vinyl chloride; chlorobenzene; bromobenzene; iodobenzene;
fluorobenzene; 1,2-dichlorobenzene; benzyl chloride; benzyl bromide; allyl bromide;
hexafluorobenzene; freon 12; halothane; methylamine; dimethylamine; trimethylamine;
ethylamine; diethylamine; triethylamine; diisopropylethylamine; aniline;
N,N-dimethylaniline; benzylamine; ethylenediamine; ethanolamine; diethanolamine;
triethanolamine; cyclohexylamine; hexamethylenetetramine; pyridine; 2-picoline;
4-dimethylaminopyridine; pyrrole; pyrrolidine; piperidine; piperazine; morpholine;
imidazole; pyrazole; indole; quinoline; isoquinoline; pyrimidine; pyrazine; purine;
pteridine; acridine; carbazole; 1,8-diazabicyclo[5.4.0]undec-7-ene; DABCO; nitromethane;
nitroethane; nitrobenzene; 2,4,6-trinitrotoluene; picric acid; nitroglycerin; glycine;
alanine; valine; leucine; isoleucine; proline; phenylalanine; tryptophan; tyrosine; serine;
threonine; cysteine; cystine; methionine; asparagine; glutamine; aspartic acid; glutamic
acid; lysine; arginine; histidine; ornithine; citrulline; taurine; beta-alanine;
gamma-aminobutyric acid; creatine; creatinine; carnitine; glucose; D-glucose; fructose;
galactose; mannose; ribose; deoxyribose; xylose; arabinose; sucrose; lactose; maltose;
trehalose; cellobiose; glucosamine; sialic acid; adenine; guanine; cytosine; thymine;
uracil; adenosine; guanosine; cytidine; thymidine; uridine; adenosine triphosphate;
adenosine monophosphate; nicotinamide adenine dinucleotide; cholesterol; testosterone;
estradiol; estrone; progesterone; cortisol; cortisone; aldosterone; prednisone;
dexamethasone; retinol; beta-carotene; lycopene; lutein; tocopherol; vitamin D3;
ergocalciferol; phylloquinone; thiamine; riboflavin; niacin; nicotinamide; pyridoxine;
biotin; folic acid; cyanocobalamin; pantothenic acid; caffeine; theobromine; theophylline;
nicotine; cocaine; morphine; codeine; heroin; quinine; strychnine; atropine; scopolamine;
ephedrine; pseudoephedrine; capsaicin; piperine; curcumin; resveratrol; quercetin;
kaempferol; catechin; epigallocatechin gallate; rutin; hesperidin; naringenin; genistein;
daidzein; eugenol; safrole; coumarin; umbelliferone; artemisinin; taxol; camptothecin;
vinblastine; colchicine; reserpine; tubocurarine; muscarine; aspirin; paracetamol;
ibuprofen; naproxen; diclofenac; ketoprofen; indomethacin; celecoxib; metamizole;
phenacetin; salbutamol; propranolol; atenolol; metoprolol; amlodipine; nifedipine;
captopril; enalapril; lisinopril; losartan; valsartan; furosemide; hydrochlorothiazide;
spironolactone; digoxin; warfarin; heparin; clopidogrel; atorvastatin; simvastatin;
rosuvastatin; metformin; glibenclamide; glimepiride; insulin; sitagliptin; pioglitazone;
penicillin G; amoxicillin; ampicillin; cephalexin; ceftriaxone; ciprofloxacin; levofloxacin;
tetracycline; doxycycline; erythromycin; azithromycin; clarithromycin; chloramphenicol;
streptomycin; gentamicin; vancomycin; rifampicin; isoniazid; ethambutol; pyrazinamide;
sulfamethoxazole; trimethoprim; metronidazole; fluconazole; ketoconazole; clotrimazole;
acyclovir; oseltamivir; zidovudine; chloroquine; hydroxychloroquine; artesunate; mefloquine;
primaquine; ivermectin; albendazole; mebendazole; praziquantel; omeprazole; lansoprazole;
ranitidine; cimetidine; loratadine; cetirizine; diphenhydramine; chlorpheniramine;
promethazine; dextromethorphan; guaifenesin; loperamide; ondansetron; metoclopramide;
domperidone; diazepam; lorazepam; alprazolam; midazolam; phenobarbital; phenytoin;
carbamazepine; valproic acid; lamotrigine; levetiracetam; gabapentin; haloperidol;
chlorpromazine; olanzapine; risperidone; clozapine; fluoxetine; sertraline; paroxetine;
citalopram; amitriptyline; imipramine; lithium carbonate; methylphenidate; amphetamine;
methamphetamine; MDMA; LSD; psilocybin; mescaline; tramadol; fentanyl; methadone; naloxone;
buprenorphine; ketamine; propofol; lidocaine; procaine; benzocaine; sildenafil; tadalafil;
finasteride; tamoxifen; letrozole; methotrexate; cisplatin; 5-fluorouracil;
cyclophosphamide; doxorubicin; imatinib; thalidomide; misoprostol; oxytocin; levothyroxine;
DDT; lindane; aldrin; dieldrin; glyphosate; atrazine; paraquat; malathion; parathion;
carbaryl; 2,4-dichlorophenoxyacetic acid; bisphenol A; dioxin; polychlorinated biphenyl;
dimethyl phthalate; diethyl phthalate; di(2-ethylhexyl) phthalate; sodium dodecyl sulfate;
EDTA; ethylenediaminetetraacetic acid; TRIS; HEPES; phenolphthalein; methyl orange; methyl
red; bromothymol blue; methylene blue; crystal violet; fluorescein; rhodamine B; eosin;
congo red; indigo; alizarin; luminol; ninhydrin; dithizone; 8-hydroxyquinoline;
dimethylglyoxime; 1,10-phenanthroline; 2,2'-bipyridine; ferrocene; triphenylphosphine;
triphenylphosphine oxide; tetramethylsilane; n-butyllithium; lithium aluminium hydride;
sodium borohydride; diborane; borane tetrahydrofuran complex; trimethyl borate; tetraethyl
orthosilicate; cyclodextrin; 18-crown-6; benzoquinone; p-benzoquinone; anthraquinone;
naphthoquinone; chloranil; DDQ; TEMPO; AIBN; benzoyl peroxide; di-tert-butyl peroxide;
N-bromosuccinimide; N-chlorosuccinimide; succinimide; phthalimide; maleic anhydride;
phthalic anhydride; succinic anhydride; caprolactam; hexamethylenediamine; vinyl acetate;
methyl methacrylate; ethyl acrylate; butyl acrylate; acrylamide; tetrafluoroethylene;
chloroprene; diphenylmethane; triphenylmethane; trityl chloride; dibenzylideneacetone;
chalcone; flavone; xanthone; stilbene; azobenzene; hydrazobenzene; phenylhydrazine;
2,4-dinitrophenylhydrazine; semicarbazide; hydroxylamine; diazomethane;
trimethylsilyldiazomethane; sodium azide; phenyl isocyanate; methyl isocyanate; toluene
diisocyanate
"""

def curated_names() -> list[str]:
    """The curated names (``;``-separated) plus the app's own offline list."""
    names = [n.strip() for n in " ".join(CURATED.split("\n")).split(";")]
    extra = []
    try:
        sys.path.insert(0, str(ROOT))
        from rsstatchem.core.chem_draw import names as nm
        extra = [c for c, _s, _smi in nm.COMMON_COMPOUNDS]
    except Exception:  # noqa: BLE001 - the builder also runs without the app importable
        pass
    seen, out = set(), []
    for n in names + extra:
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


class Client:
    """Polite PUG-REST client: rate limit, back-off on 503 / red throttling, answer cache."""

    def __init__(self, rate: float, cache: Path | None):
        self.interval = 1.0 / max(0.1, min(rate, 4.5))
        self.cache = cache
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
        self.last = 0.0
        self.requests = 0
        try:
            import certifi
            self.ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            self.ctx = ssl.create_default_context()

    def _cached(self, key: str) -> Path | None:
        if self.cache is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:120]
        return self.cache / f"{safe}.json"

    def request(self, path: str, data: dict | None = None, key: str = "") -> dict | None:
        cpath = self._cached(key) if key else None
        if cpath is not None and cpath.exists():
            return json.loads(cpath.read_text(encoding="utf-8"))
        body = urllib.parse.urlencode(data).encode() if data else None
        delay = 2.0
        for _attempt in range(8):
            wait = self.last + self.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.requests += 1
            req = urllib.request.Request(f"{PUG}/{path}", data=body,
                                         headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=60, context=self.ctx) as resp:
                    throttle = resp.headers.get("X-Throttling-Control", "")
                    raw = resp.read()
                if "Red" in throttle or "Black" in throttle:
                    print(f"  server asks to slow down: {throttle}", flush=True)
                    time.sleep(30)
                elif "Yellow" in throttle:
                    time.sleep(2)
                result = json.loads(raw.decode("utf-8"))
                if cpath is not None:
                    cpath.write_text(json.dumps(result), encoding="utf-8")
                return result
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    result = {}
                    if cpath is not None:
                        cpath.write_text("{}", encoding="utf-8")
                    return result
                if exc.code in (429, 500, 502, 503, 504):
                    print(f"  HTTP {exc.code}; retrying in {delay:.0f} s", flush=True)
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                if exc.code == 400:
                    return {}
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"  network error ({exc}); retrying in {delay:.0f} s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
        print("  giving up on this request", flush=True)
        return None


def fetch_properties(client: Client, cids: list[int]) -> list[dict]:
    res = client.request("compound/cid/property/" + PROPS + "/JSON",
                         {"cid": ",".join(map(str, cids))},
                         key=f"props_{cids[0]}_{cids[-1]}_{len(cids)}")
    if not res:
        return []
    return res.get("PropertyTable", {}).get("Properties", [])


def resolve_names(client: Client, names: list[str]) -> dict[str, int]:
    out = {}
    for k, name in enumerate(names):
        res = client.request("compound/name/" + urllib.parse.quote(name, safe="") +
                             "/cids/JSON", key="name_" + name.lower())
        cids = (res or {}).get("IdentifierList", {}).get("CID", [])
        if cids:
            out[name] = int(cids[0])
        if (k + 1) % 50 == 0:
            print(f"  names resolved: {k + 1}/{len(names)} ({len(out)} found)", flush=True)
    return out


def clean_row(p: dict) -> tuple[str, str, str, str] | None:
    key = str(p.get("InChIKey") or "")
    if len(key) != 27:
        return None
    iupac = str(p.get("IUPACName") or "").strip()
    title = str(p.get("Title") or "").strip()
    smiles = str(p.get("SMILES") or p.get("IsomericSMILES") or "").strip()
    if len(iupac) > MAX_IUPAC:
        iupac = ""
    if len(smiles) > MAX_SMILES:
        smiles = ""
    if re.fullmatch(r"(CID|SID|NSC|AC1|CHEMBL|SCHEMBL|ZINC|DTXSID|UNII)[\s\-:]?\w*", title,
                    re.I) or re.fullmatch(r"[A-Z0-9\-]{6,}", title):
        title = ""                                 # registry numbers are not names
    if title.lower() == iupac.lower():
        title = ""
    if not (iupac or title):
        return None
    for s in (iupac, title, smiles):
        if "\t" in s or "\n" in s:
            return None
    return key, iupac, title, smiles


def encode(rows: list[tuple], header: str) -> bytes:
    text = header + "".join("\t".join(r) + "\n" for r in rows)
    return lzma.compress(text.encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-cid", type=int, default=120000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--rate", type=float, default=3.0, help="requests per second (<= 4.5)")
    ap.add_argument("--max-mb", type=float, default=4.5)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    client = Client(args.rate, args.cache)

    names = curated_names()
    print(f"Resolving {len(names)} curated names...", flush=True)
    curated = resolve_names(client, names)
    curated_cids = set(curated.values())
    print(f"  {len(curated_cids)} curated compounds", flush=True)

    props: dict[int, dict] = {}
    extra = sorted(c for c in curated_cids if c > args.max_cid)
    for k in range(0, len(extra), args.batch):
        for p in fetch_properties(client, extra[k:k + args.batch]):
            props[int(p["CID"])] = p
    print(f"Fetching CIDs 1..{args.max_cid} in batches of {args.batch}...", flush=True)
    t0 = time.monotonic()
    for start in range(1, args.max_cid + 1, args.batch):
        cids = list(range(start, min(start + args.batch, args.max_cid + 1)))
        for p in fetch_properties(client, cids):
            props[int(p["CID"])] = p
        if (start // args.batch) % 20 == 0:
            print(f"  up to CID {cids[-1]}: {len(props)} records, {client.requests} requests, "
                  f"{time.monotonic() - t0:.0f} s", flush=True)

    rows_by_key: dict[str, tuple[int, tuple]] = {}
    for cid in sorted(props):
        row = clean_row(props[cid])
        if row is None:
            continue
        rank = 0 if cid in curated_cids else cid           # curated first, then low CIDs
        old = rows_by_key.get(row[0])
        if old is None or rank < old[0]:
            rows_by_key[row[0]] = (rank, row)

    def build(cutoff: int) -> tuple[bytes, int]:
        rows = sorted(r for rank, r in rows_by_key.values() if rank <= cutoff)
        header = (f"# RS-StatChem offline name database v1 ({len(rows)} compounds)\n"
                  f"# Source: PubChem PUG-REST compound records (public domain), "
                  f"generated {date.today().isoformat()} by scripts/build_name_db.py\n"
                  "# InChIKey\tIUPAC name\tcommon name (PubChem title)\tSMILES\n")
        return encode(rows, header), len(rows)

    cutoff = max(args.max_cid, max(curated_cids, default=0))
    data, n = build(cutoff)
    limit = int(args.max_mb * 1024 * 1024)
    lo, hi = 1, cutoff
    if len(data) > limit:
        while hi - lo > 500:                            # largest CID cutoff that fits
            mid = (lo + hi) // 2
            d, _ = build(mid)
            if len(d) <= limit:
                lo = mid
            else:
                hi = mid
        data, n = build(lo)
        print(f"Trimmed to CIDs <= {lo} to stay under {args.max_mb} MB", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"Wrote {args.out} : {n} compounds, {len(data) / 1e6:.2f} MB "
          f"({client.requests} requests)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
