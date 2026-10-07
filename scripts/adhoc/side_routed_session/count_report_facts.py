import re

import polars as pl

reports = pl.read_csv("data/raw/train.csv", columns=["Report"])["Report"].fill_null("").str.to_lowercase().to_list()
total = len(reports)
NEAR = 90


def near(first, second):
    return re.compile(rf"(?:{first})[^.;\n]{{0,{NEAR}}}(?:{second})|(?:{second})[^.;\n]{{0,{NEAR}}}(?:{first})", re.S)


EDEMA = r"edema|oedem|œdème|ödem|οίδημα|οιδημα|edem|оток|contusi|kontusi|bone bruise|kontüzyon|κάκωσ|kontuzij|контузи"
FEMUR = r"femoral condyle|cóndilo|condilo|kondyl|condyle|kondil|κόνδυλ|κονδυλ|кондил|fémur|femur|femoral"
TIBIA = r"tibial plateau|platillo|meseta tibial|tibiakopf|tibiaplateau|plateau|plato|tibia|κνημιαί|κνημια|platoa|плато|тибиал"
PATELLA = r"patella|rótula|rotula|kniescheibe|rotule|knieschijf|επιγονατίδα|επιγονατιδ|iver|пател"
CARTILAGE = r"cartilag|cartílago|kraakbeen|knorpel|kıkırdak|χόνδρ|χονδρ|hrskavic|хрущял|chondr|kondr|condr|хондр"
QUADRICEPS = r"quadriceps|cuádriceps|cuadriceps|kuadriseps|τετρακέφαλ|τετρακεφαλ|kvadriceps|квадрицепс"
PATELLAR_TENDON = r"patellar tendon|tendón rotuliano|tendon rotuliano|patellarsehne|patellasehne|tendon rotulien|patellapees|patellar tendon|ligamentum patellae|patellar ligament|επιγονατιδικ[^ ]* τένοντ|patelarn[^ ]* (?:tetiv|ligament)|patellar tendon"
TENDON_FINDING = r"tendin|tendon[^.;]{0,40}(?:tear|rupt|thick)|rupt|desgarr|riss|déchir|scheur|yırtı|ρήξη|ρηξη|ruptur|руптур|kalınlaş|verdick|engros"
MENISCUS = r"menisc|menisk|μηνίσκ|μηνισκ|мениск"
ROOT = r"\broot\b|raíz|raiz|wurzel|racine|wortel|kök|ρίζα|ριζα|korijen|корен"
PERIPHERAL = r"periph|perif|kapsel|capsul|kapsül|meniscocapsular|red zone|ramp|περιφερ|периф|kapsul"
FREE_EDGE = r"free edge|inner edge|borde libre|freien rand|freier rand|bord libre|vrije rand|serbest kenar|ελεύθερο χείλος|slobodn[^ ]* rub|свободн[^ ]* ръб"
ACL = r"\bacl\b|\blca\b|\bvkb\b|vorderes kreuzband|vorderen kreuzband|voorste kruisband|ön çapraz|πρόσθι[^ ]* χιαστ|prednj[^ ]* križn|предна[^ ]* кръстн|anterior cruciate|cruzado anterior|croisé antérieur"
MCL = r"\bmcl\b|\blcm\b|\bllm\b|innenband|medial collateral|colateral medial|collatéral médial|mediale collaterale|iç yan bağ|έσω πλάγι|medijaln[^ ]* kolateral|медиал[^ ]* колатерал|\blli\b"
PROXIMAL = r"proximal|femoral (?:insertion|attachment|origin)|femoral|ursprung|origin|orijin|έκφυση|εκφυση|polazišt|залавно"
DISTAL = r"distal|tibial (?:insertion|attachment)|tibial|ansatz|insertion|inserción|yapışma|κατάφυση|καταφυση|hvatišt"
ANTERIOR = r"anterior|vorder|antérieur|voorste|ön |πρόσθι|προσθι|prednj|предн"
POSTERIOR = r"posterior|hinter|postérieur|achterste|arka |οπίσθι|οπισθι|stražnj|задн"
FRACTURE = r"fractur|fraktur|kırık|κάταγμα|καταγμα|prijelom|фрактур|fx"

facts = {
    "marrow edema or contusion mentioned": re.compile(EDEMA),
    "  edema with a femur word nearby": near(EDEMA, FEMUR),
    "  edema with a tibia word nearby": near(EDEMA, TIBIA),
    "  edema with a kneecap word nearby": near(EDEMA, PATELLA),
    "cartilage mentioned": re.compile(CARTILAGE),
    "  cartilage with a femur word nearby": near(CARTILAGE, FEMUR),
    "  cartilage with a tibia word nearby": near(CARTILAGE, TIBIA),
    "  cartilage with anterior nearby": near(CARTILAGE, ANTERIOR),
    "  cartilage with posterior nearby": near(CARTILAGE, POSTERIOR),
    "quadriceps tendon named": re.compile(QUADRICEPS),
    "patellar tendon named": re.compile(PATELLAR_TENDON),
    "meniscus with a root word nearby": near(MENISCUS, ROOT),
    "meniscus with a peripheral or capsular word nearby": near(MENISCUS, PERIPHERAL),
    "meniscus with a free-edge word nearby": near(MENISCUS, FREE_EDGE),
    "ACL with proximal or femoral nearby": near(ACL, PROXIMAL),
    "ACL with distal or tibial nearby": near(ACL, DISTAL),
    "MCL with proximal or femoral nearby": near(MCL, PROXIMAL),
    "MCL with distal or tibial nearby": near(MCL, DISTAL),
    "fracture mentioned": re.compile(FRACTURE),
    "  fracture with a femur word nearby": near(FRACTURE, FEMUR),
    "  fracture with a tibia word nearby": near(FRACTURE, TIBIA),
    "  fracture with a kneecap word nearby": near(FRACTURE, PATELLA),
}
print(total, "reports")
for name, pattern in facts.items():
    count = sum(bool(pattern.search(report)) for report in reports)
    print(f"{name:52} {count:5}  {100 * count / total:5.1f}%")
