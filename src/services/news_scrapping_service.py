import os
from utils.file_utils import write_file
from utils.text_utils import format_filename

from goose3 import Goose
from scrapling.fetchers import StealthySession

class NewsScrappingService:
    def __init__(self):
        pass

    def get_article_text(self, url: str, output_dir: str):
        # El funcionamiento correcto sería extraer esto a una clase "Scrapper", "Spider" o similar.
        with StealthySession(headless=True, solve_cloudflare=True) as session:
            page = session.fetch(url, google_search=False, disable_resources=True)
        
        g = Goose()
        article = g.extract(url="", raw_html=page.html_content)
        file_name = format_filename(article.title) + ".txt"
        file_path = output_dir + "/" + file_name
        print(f"Guardada transcripción de la noticia como {file_name}")

        write_file(file_path, article.cleaned_text)