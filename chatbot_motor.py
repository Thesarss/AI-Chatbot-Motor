import json

# ==============================
# Load Dataset
# ==============================
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)


# ==============================
# Fungsi Chatbot
# ==============================
def cari_solusi(pertanyaan):
    pertanyaan = pertanyaan.lower()
    for item in dataset:
        if any(kata in pertanyaan for kata in item["gejala"].lower().split()):
            return {
                "gejala": item["gejala"],
                "penyebab": item["kemungkinan_penyebab"],
                "solusi": item["solusi"]
            }
    return None


# ==============================
# Main Program
# ==============================
print("🤖 Chatbot Otomotif - Permasalahan Motor")
print("Ketik 'exit' untuk keluar.\n")

while True:
    user_input = input("Kamu: ")
    if user_input.lower() == "exit":
        print("Chatbot: Sampai jumpa, semoga motor kamu sehat selalu 🚀")
        break

    hasil = cari_solusi(user_input)

    if hasil:
        print("\nChatbot: Saya menemukan gejala yang mirip dengan masalahmu:")
        print("👉 Gejala:", hasil["gejala"])
        print("🔎 Kemungkinan Penyebab:", ", ".join(hasil["penyebab"]))
        print("🛠 Solusi:", ", ".join(hasil["solusi"]))
        print()
    else:
        print("\nChatbot: Maaf, saya belum punya data untuk masalah itu. 🚧\n")
