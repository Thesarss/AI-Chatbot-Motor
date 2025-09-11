# File: chat_with_ai.py (Versi Konsolidasi Final)

import uuid
# ✅ Pastikan impornya benar
from main_enhanced_ai import EnhancedMotorcycleAssistant

def print_welcome():
    """Mencetak pesan selamat datang."""
    print("\n" + "="*60)
    print("🔧 SELAMAT DATANG DI BENGKELAI ASSISTANT (v4.0) 🔧")
    print("="*60)
    print("AI Assistant untuk membantu masalah motor Anda.")
    print("Ketik 'exit' atau 'keluar' untuk mengakhiri chat.")
    print("Ketik 'stats' untuk melihat statistik sesi.")
    print("Ketik 'reset' untuk memulai sesi baru.")
    print("="*60 + "\n")

def print_stats(assistant, session_id):
    """Mencetak statistik sesi."""
    try:
        stats = assistant.get_session_stats(session_id)
        print("\n📊 STATISTIK SESI:")
        for key, value in stats.items():
            print(f"   - {key.replace('_', ' ').title()}: {value}")
        print()
    except Exception as e:
        print(f"❌ Error mendapatkan statistik: {str(e)}\n")

def main():
    """Fungsi chat utama."""
    print_welcome()
    
    try:
        print("🔄 Menginisialisasi AI Assistant...")
        # ✅ Memanggil kelas yang sudah dikonsolidasi
        assistant = EnhancedMotorcycleAssistant(
            model_path="./bengkelAI-Gemma-LORA-final"
        )
        print("✅ AI Assistant berhasil diinisialisasi!\n")
    except Exception as e:
        print(f"❌ Gagal menginisialisasi AI: {e}")
        print("💡 Pastikan Anda sudah login ke Hugging Face dan path model sudah benar.")
        return
    
    session_id = f"chat_{uuid.uuid4().hex[:6]}"
    print(f"🆔 Session ID: {session_id}")
    print("\n💬 Silakan mulai percakapan:")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n👤 Anda: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'keluar']:
                print("\n👋 Terima kasih telah menggunakan BengkelAI!")
                break
            elif user_input.lower() == 'stats':
                print_stats(assistant, session_id)
                continue
            elif user_input.lower() == 'reset':
                session_id = f"chat_{uuid.uuid4().hex[:6]}"
                assistant.context_manager.active_sessions.pop(session_id, None) # Hapus konteks lama
                print(f"\n🔄 Sesi baru dimulai!")
                print(f"🆔 Session ID: {session_id}")
                continue
            elif not user_input:
                continue
            
            print("\n🤖 AI sedang berpikir...")
            response = assistant.process_message(user_input, session_id)
            
            print(f"\n🤖 BengkelAI: {response}")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat dihentikan. Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Terjadi Error: {e}")

if __name__ == "__main__":
    main()