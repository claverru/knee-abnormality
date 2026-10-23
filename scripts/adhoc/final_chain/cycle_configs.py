import re
from pathlib import Path

SRC = Path("/home/claudio/src")
TABLE = "  teacher_table: ../data/labels/prepared/teachers2-kaggle-x4-tau085-tta/kaggle_teacher_table.parquet\n"
KAGGLE_PATHS = (
    "  anatomy_targets: ../data/labels/prepared/anatomy-mrannotator-sample550-joints\n"
    "  meniscus_targets: ../data/labels/prepared/soft-tissue-nnunet-knee-dess265\n"
)
FAMILIES = {
    "convnext": ("knee-abnormality-sync-free", "convnext.yaml", "convnext-teachers2-joint-v2.yaml"),
    "mambavision": ("knee-abnormality-sync-free", "mambavision.yaml", "mambavision-teachers2-joint-v2.yaml"),
    "resnet50": ("knee-abnormality-sync-free", "resnet50.yaml", "resnet50-teachers2-joint-v2.yaml"),
    "focalnet": ("knee-abnormality-teachers2", None, "focalnet-teachers2-joint-v2.yaml"),
    "efficientvit": ("knee-abnormality-effvit", "efficientvit.yaml", "efficientvit-teachers2-joint.yaml"),
}


def with_share(text: str, name: str) -> str:
    text = re.sub(r"^name: .*$", f"name: {name}", text, count=1, flags=re.M)
    text = re.sub(r"\ndistillation:\n  teacher_share: [0-9.]+\n", "\n", text)
    text, count = re.subn(r"^(  meniscus_targets: .*\n)", r"\1" + TABLE.replace("\\", "\\\\") + "\ndistillation:\n  teacher_share: 0.75\n", text, count=1, flags=re.M)
    assert count == 1, name
    return text


for family, (tree, base, joint) in FAMILIES.items():
    configs = SRC / tree / "configs"
    joint_text = (configs / joint).read_text()
    if base is None:
        kaggle = re.sub(r"\nextra_training:\n(  .*\n)+", "\n", joint_text)
        kaggle = re.sub(r"  anatomy_targets: .*\n  meniscus_targets: .*\n", KAGGLE_PATHS, kaggle, count=1)
    else:
        kaggle = (configs / base).read_text()
    assert "extra_training" not in kaggle and "external" not in kaggle, family
    (configs / f"{family}-share075-finetune.yaml").write_text(with_share(kaggle, f"{family}-share075-finetune"))
    (configs / f"{family}-share075-joint.yaml").write_text(with_share(joint_text, f"{family}-share075-joint"))
    print("wrote", tree, family)
