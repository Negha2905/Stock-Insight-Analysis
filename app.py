from frontend import app
from database import create_users_table, init_db,create_watchlist_table

def main():
    create_users_table()
    create_watchlist_table()
    init_db()  # Ensure the user_content table is created
    app()

if __name__ == "__main__":
    main()

