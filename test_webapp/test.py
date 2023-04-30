import requests
from bs4 import BeautifulSoup

def get_balance_sheet(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/balance-sheet/"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find_all('table')[0]
    rows = table.find_all('tr')

    # Extract the column headings from the first row
    headings = [th.get_text().strip() for th in rows[0].find_all('th')]

    # Extract the balance sheet data for each row
    balance_sheet_data = []
    for row in rows[1:]:
        data = [td.get_text().strip() for td in row.find_all('td')]
        balance_sheet_data.append(data)

    # Combine the column headings and balance sheet data into a dictionary
    balance_sheet_dict = {}
    for i, heading in enumerate(headings):
        column_data = [row[i] for row in balance_sheet_data]
        balance_sheet_dict[heading] = column_data

    return balance_sheet_dict


balance_sheet = get_balance_sheet("AAPL")
print(balance_sheet)
