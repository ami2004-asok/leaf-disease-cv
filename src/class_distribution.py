from pathlib import Path

ROOT = Path("data/raw")

print("\nClass Distribution\n")

for class_dir in ROOT.iterdir():

    if class_dir.is_dir():

        count = len(
            [
                file
                for file in class_dir.glob("*")
                if file.is_file()
            ]
        )

        print(
            f"{class_dir.name:<15} : {count}"
        )