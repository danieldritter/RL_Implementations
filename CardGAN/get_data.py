import requests


if __name__ == "__main__":
    # From Scryfall website
    number_of_pages = 1460

    # TODO: Loop through all pages and pull their image_uris. Then Use
    # Image uris to pull artwork and store it as a dataset. Then you can
    # process it normally 
    payload = {'page': '2'}
    data = requests.get("https://api.scryfall.com/cards", params=payload)
    print(data.json()['data'][0])
