"""Script utilitário para limpar o diretório storage/ e o histórico de eventos."""

import shutil
import sys
from pathlib import Path


def clean_storage(interactive: bool = True):
    root_dir = Path(__file__).resolve().parent
    storage_dir = root_dir / "storage"
    events_dir = storage_dir / "events"
    events_log = storage_dir / "events.jsonl"
    events_bak = storage_dir / "events.jsonl.bak"

    print("=" * 60)
    print(" Limpeza de Storage e Histórico de Eventos - Cam-Security")
    print("=" * 60)

    if not storage_dir.exists():
        print(f"[INFO] Diretório '{storage_dir}' não encontrado. Nada a limpar.")
        return

    # Contagem de itens para exibir ao usuário
    total_event_folders = 0
    if events_dir.exists():
        total_event_folders = sum(1 for p in events_dir.iterdir() if p.is_dir())

    log_size = events_log.stat().st_size if events_log.exists() else 0

    print(f"Alvos para limpeza:")
    print(f" - Pastas de eventos gravados ({events_dir}): {total_event_folders} pastas")
    print(f" - Histórico JSONL ({events_log}): {log_size} bytes")
    if events_bak.exists():
        print(f" - Backup JSONL ({events_bak}): presente")

    if interactive:
        confirm = input("\nTem certeza que deseja apagar todos os eventos e gravações? (s/N): ").strip().lower()
        if confirm not in ("s", "sim", "y", "yes"):
            print("[CANCELADO] Operação cancelada pelo usuário.")
            return

    # Remover todo o conteúdo de storage/events/
    if events_dir.exists():
        deleted_count = 0
        for item in list(events_dir.iterdir()):
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"[ERRO] Falha ao apagar '{item.name}': {e}")
        print(f"[OK] Diretório '{events_dir}' limpo ({deleted_count} itens excluídos).")
    else:
        events_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Diretório '{events_dir}' criado.")

    # Remover logs
    if events_log.exists():
        try:
            events_log.unlink()
            print(f"[OK] Arquivo '{events_log.name}' removido.")
        except Exception as e:
            print(f"[ERRO] Falha ao remover '{events_log}': {e}")

    if events_bak.exists():
        try:
            events_bak.unlink()
            print(f"[OK] Arquivo '{events_bak.name}' removido.")
        except Exception as e:
            print(f"[ERRO] Falha ao remover '{events_bak}': {e}")

    print("\n[SUCESSO] Storage e histórico limpos com sucesso!")


if __name__ == "__main__":
    is_interactive = "--force" not in sys.argv and "-f" not in sys.argv
    clean_storage(interactive=is_interactive)
