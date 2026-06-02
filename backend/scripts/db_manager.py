import os
import subprocess

# Settings
CONTAINER_NAME = "trading_management-db"
BACKUP_FILE = "db_backup.gz"
BACKUP_PATH = os.path.join(os.path.dirname(__file__), "..", "backups", BACKUP_FILE)


def ensure_backup_dir():
    backup_dir = os.path.dirname(BACKUP_PATH)
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)


def backup_db():
    """Creates a compressed archive of the entire MongoDB database."""
    ensure_backup_dir()
    print(f"📦 Starting backup of {CONTAINER_NAME}...")

    # We use mongodump via docker and stream it to a local file
    try:
        cmd = f"docker exec {CONTAINER_NAME} mongodump --archive --gzip"
        with open(BACKUP_PATH, "wb") as f:
            subprocess.run(cmd, shell=True, check=True, stdout=f)
        print(f"✅ Backup successful! Saved to: {BACKUP_PATH}")
        print("💡 You can now commit this file to GitHub to sync your data.")
    except Exception as e:
        print(f"❌ Backup failed: {e}")


def restore_db():
    """Restores the database from the backup file."""
    if not os.path.exists(BACKUP_PATH):
        print(f"❌ No backup file found at {BACKUP_PATH}")
        return

    print(f"🔄 Restoring database from {BACKUP_FILE}...")
    try:
        cmd = f"docker exec -i {CONTAINER_NAME} mongorestore --archive --gzip --drop"
        with open(BACKUP_PATH, "rb") as f:
            subprocess.run(cmd, shell=True, check=True, stdin=f)
        print("✅ Restore successful! Your database is now up to date.")
    except Exception as e:
        print(f"❌ Restore failed: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python db_manager.py [backup|restore]")
    elif sys.argv[1] == "backup":
        backup_db()
    elif sys.argv[1] == "restore":
        restore_db()
