import os

# format -> grupNo_yazlab2_kaynakkod.txt
output_filename = "57_yazlab2_kaynakkod.txt"

# koda dahil edilmemesi gereken gereksiz klasörler
exclude_dirs = ["venv", "__pycache__", ".pytest_cache", ".git", "data", "images", "output"]

with open(output_filename, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            # sadece python scriptleri ve config dosyası alınır
            if file.endswith(".py") or file.endswith(".yaml"):
                
                filepath = os.path.join(root, file)
                outfile.write(f"\n{'='*60}\n")
                outfile.write(f"DOSYA: {filepath}\n")
                outfile.write(f"{'='*60}\n\n")
                
                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"# Dosya okunamadi: {e}\n")

print(f"İşlem tamam! Tüm kodlar {output_filename} dosyasında birleştirildi.")