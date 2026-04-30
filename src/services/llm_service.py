from llm import LLMClient
from utils.file_utils import read_file, read_json_file, write_json_file

import re
import requests
import os

class LLMService:
    def __init__(self, llm_model: LLMClient):
        self.llm_model = llm_model

    def _search_mentions(self, texto, palabras_clave, contexto=500):
        palabras_regex = r"\b(" + "|".join(re.escape(p) + r"s?" for p in palabras_clave) + r")\b"
        pattern = re.compile(palabras_regex, re.IGNORECASE)
        
        vistos = set()
        resultados = []
        for m in pattern.finditer(texto):
            fragmento = texto[max(0, m.start() - contexto):min(len(texto), m.end() + contexto)].strip()
            if fragmento not in vistos:
                vistos.add(fragmento)
                resultados.append({
                    "mencion": m.group(),
                    "texto": fragmento
                })
        
        return resultados
    
    def chunking(self, text, chunk_size=850, overlap=150):
        """
        Crea chunks a partir de finales de frases para obtener texto reducido, con algo de overlap.
        """

        sentences = re.split(r'(?<=[.!?])\s+', text) # Busca caracteres de final de línea.
    
        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            words = len(sentence.split())
            if current_len + words > chunk_size and current:
                chunks.append(" ".join(current))
                overlap_text = " ".join(current).split()[-overlap:] # Mantener las últimas palabras como "overlap".
                current = [" ".join(overlap_text)]
                current_len = len(overlap_text)
            current.append(sentence)
            current_len += words

        if current:
            chunks.append(" ".join(current))

        return chunks
    
    def _save_results(self, salida_json, origen_txt, fragmentos):
        clave = os.path.splitext(os.path.basename(origen_txt))[0].lower() 

        data = read_json_file(salida_json) or {}
        data[clave] = list(dict.fromkeys(data.get(clave, []) + fragmentos))

        write_json_file(salida_json, data)
        print(f"Resultados añadidos a '{salida_json}' bajo la clave '{clave}'.")

    def search_relevant_fragments(self, fragments, keywords, headers):
        keywords_str = ", ".join(keywords)

        

        system_prompt = """
            Analiza el siguiente texto y extrae TODAS las menciones de estas palabras o expresiones.

            Para cada mención encontrada:
            1. Cita el fragmento exacto donde aparece (entre 1 y 3 frases, las suficientes para entender el contexto).
            2. Indica la palabra o expresión que ha activado la búsqueda.
            3. Si la mención es ambigua o indirecta (por ejemplo, un pronombre que se refiere a esa persona), inclúyela igualmente y señálalo.

            Formato de salida:
            ---
            Mención [número]
            Palabra clave: [palabra encontrada]
            Fragmento: "[texto extraído]"
            Nota (opcional): [si hay algo que aclarar sobre el contexto]
            ---
        """
        
        system_prompt = (
            "Eres un asistente especializado en recopilar el contexto de textos."
            "Para separar menciones en contextos distintos, utiliza el siguiente separador: '\n=====\n'"
            "Se te proporcionará un fragmento de una transcripción de vídeo y una lista de entidades (personas, empresas u organizaciones) a buscar."
            "Incluye las menciones de estas palabras, directas o implícitas (por ejemplo, pronombres, apodos, o alusiones claras con el resto del contexto), incluso si varias oraciones expresan ideas similares."
            "Incluye frases alrededor de las menciones si esclarecen el mismo contexto."
            "Tu respuesta debe empezar tras un punto y acabar en un punto."
            "Si no hay ninguna mención, responde exactamente 'NINGUNO'."
            "Devuelve únicamente el resultado sin explicaciones ni comentarios."
        )

        results = []
        for idx, frag in enumerate(fragments, 1):
            try:
                print(f"Buscando fragmentos relevantes en el chunk [{ idx }/{ len(fragments) }].")
                user_prompt = (
                    f"Entidades a buscar: {keywords_str}\n"
                    f"Fragmento de transcripción: \n{frag}"
                )

                result = self.llm_model.call(user_prompt, system_prompt, headers)
                if result: 
                    results.extend(result)
            except Exception as e:
                print(f"Error procesando fragmento {idx}: {e}")
                continue

        if len(results) < 1:
            print("No se ha encontrado ninguna mención de las palabras clave.")
        
        return results

    def check_relevant_fragments(self, fragments, keywords, headers):
        keywords_str = ", ".join(keywords)
        system_prompt = (
            "Eres un asistente especializado en análisis de transcripciones."
            "Se te proporcionará un fragmento de una transcripción de vídeo y una lista de entidades (personas, empresas u organizaciones) a buscar."
            "Tu tarea es identificar todas las menciones de esas entidades en el fragmento, incluyendo referencias indirectas o implícitas (por ejemplo, pronombres, apodos, o alusiones claras)."
            "Si una entidad no aparece en el fragmento, no la menciones."
            "Si no hay ninguna mención relevante en el fragmento, responde únicamente con: 'NINGUNO'."
            "Sé conciso y no añadas información que no esté en el texto."
        )

        results = []
        for idx, frag in enumerate(fragments, 1):
            try:
                print(f"Analizando contexto en el fragmento [{ idx }/{ len(fragments) }].")
                user_prompt = (
                    f"Entidades a buscar: {keywords_str}\n"
                    f"Fragmento de transcripción: \n{frag}"
                )

                result = self.llm_model.call(user_prompt, system_prompt, headers)
                if result: 
                    results.extend(result)
            except Exception as e:
                print(f"Error procesando fragmento {idx}: {e}")
                continue

        if len(results) < 1:
            print("No se ha encontrado ninguna mención de las palabras clave.")
        
        return results
    
    def analize_transcription(self, input_path, keywords, json_output_path, contexto=500):
        print(f"\nProcesando archivo: { os.path.basename(input_path) }")

        texto = read_file(input_path)

        chunks = self.chunking(texto)
        print(f"El texto ha sido dividido en { len(chunks) } chunks.")

        headers = { "Content-Type": "application/json" }
        relevant_fragments = self.search_relevant_fragments(chunks, keywords,  headers)
        print(f"Fragmentos relevantes encontrados: { len(relevant_fragments) }")

        checked_fragments = self.check_relevant_fragments(relevant_fragments, keywords, headers)
        print(f"Fragmentos finales: { len(checked_fragments) }")

        if checked_fragments:
            self._save_results(json_output_path, input_path, checked_fragments)

if __name__ == "__main__":
    client = LLMClient("http://localhost:1234/v1/chat/completions", "qwen3-8b")
    service = LLMService(client)
    fragments = [
        "El que fuera comisionista del caso Coldo ha tratado en todo momento de involucrar al presidente del gobierno en esa trama corrupta y en una supuesta financiación irregular del PSOE. Ha asegurado que llevó hasta 250.000 euros enó porque necesitaba una persona de confianza que hiciese de intermediario con empresas que recibían obras, contratos de obras amañados. La tarea de Aldama sería conseguir efectivo para convertirlo en donaciones para el partido, para el PSOE. También, sin pruebas, ha tratado de implicar al presidente Pedro Sánchez y lo ha situado como número uno de la presunta organización criminal porque dice que el presidente del gobierno lo sabía todo y le agradeció lo que estaba haciendo por ellos. Aldama también ha disparado contra su mujer, contra Begoña Gómez. Ha dicho que intenta quedarse de forma ilegal con terrenos en el centro de Madrid. Aldama apunta en su confesión a lo más alto. Si hay una jerarquía en este caso, y que yo obviamente estoy en la banda organizada, criminal, el señor presidente del gobierno Pedro Sánchez está en el escalafón 1. Involucra a Pedro Sánchez varias veces, pero dice que es lo que le cuenta otro de los acusados, Coldo García. Y me ha dicho que el presidente todo lo que hacíamos lo tenía tanto poder entre ministros y presidentes de comunidades autónomas no era por ser asesor de Ábalos. Asegura que Coldo presumía de su influencia sobre Sánchez. Admite que el exministro y su asesor le ficharon para mediar con las constructoras que presuntamente conseguían adjudicaciones a cambio de mordidas. Ibas a ser tú el nexo, porque las empresas tienen que pagar en efectivo. El empresario cree que Ábalos y Coldo no se quedaban con todo. Ellos siempre me han dicho que parte de ese dinero iba para la financiación del Partido Socialista. Él entregaba el dinero frecuentemente en el ministerio, no pasaba por el control de seguridad. Si llevaba entre 50 y 60.000 euros, lo llevaba en un sobre. Hasta 250.000 euros lo llevaba en la mochila. Resario ha explicado que pagaba 10.000 euros al mes a Ábalos y el alquiler de la novia del ministro. Y una vez, prostitutas. Incluso relata una vez que llegó a las manos con Coldo. Después de una relación de amistad, dice, en la que le llamaba para todo. Y en el juicio del caso Kitschen ha declarado hoy el principal investigador del caso Gürtel y de los papeles de Bárcenas. Manuel Morocho ha contado que recibió presiones de sus superiores para quitar de sus informes el nombre de Rajoy y ha asegurado que la Kitschen fue una operación sin autorización judicial. El inspector principal que investigó Gürtel lo tiene claro. Había una operación policial sin contar con autorización judicial sobre Luis Bárcenas y su entorno. Ha contado todo tipo de presiones de sus jefes policiales, primero para desautorizar los papeles de Bárcenas. A saber Bárcenas por qué los ha hecho, que era una ideación de Bárcenas. Luego, para no mencionar a Mariano Rajoy en sus informes. Se individualizó expresamente que esa persona no saliera. O para sencillamente que cambiara sus conclusiones. Y dice que intentaron quitárselo de en medio ofreciéndole destinos mejor pagados en el extranjero. El inspector también dice que uno de sus jefes reconoció abiertamente que pasaba al PP documentos importantes de su investigación. Cuando se le ha mostrado el material presuntamente robado a vamos en directo hasta Zaragoza. Carmen Ruiz ya ha terminado. Así es, hace poco más de media hora que ha concluido en este hemiciclo esa votación que ha permitido a Jorge Azcón ser elegido por segunda vez como presidente de Aragón. Ha obtenido los votos de 39 de los 67 diputados de esta Cámara, uno menos de los que estaba previsto por encontrarse una diputada de Vox de baja por enfermedad. A lo largo de más de cinco horas de discursos que hemos escuchado esta mañana, la oposición ha criticado con dureza ese acuerdo entre PP y Vox. Dicen que abre las puertas a la discriminación, a la desigualdad y que no es legal. Por su parte, PP y Vox lo han defendido, consideran que ese arraigo sí que hace que se enmarque dentro de la legalidad y que no discrimina a nadie. Por su parte, también el futuro vicepresidente del gobierno de Aragón, el líder de Vox en la comunidad, ha dicho que no tiene que pedir perdón por esa prioridad nacional y por decir que tienen que ir primero quienes respetan nuestra cultura y nuestra civilización y que Aragón no puede seguir expulsando jóvenes mientras trae mano de obra precaria. Jorge Azcón tomará posesión por segunda vez como presidente de Aragón mañana a las 5 de la tarde en este mismo Palacio de la Aljafería. Trump quiere que su rostro aparezca también en los pasaportes. Son estos que se van a emitir por los 250 años de la independencia de Estados Unidos. Su cara en los pasaportes y también en monedas conmemorativas. Es el último ejemplo de la megalomanía que ha llevado a Trump a poner su nombre a instituciones oficiales, a documentos y hasta una flota de barcos militares. Su rostro donde antes había una bandera y un águila. Donald Trump se convertirá en el primer presidente de Estados Unidos en estampar su cara en los pasaportes del país. Serán ediciones limitadas con motivo del 250 aniversario de la independencia estadounidense. Un gesto inaudito, pero no el único. También aparecerá en varios tipos de monedas conmemorativas y su firma se estampará en los nuevos billetes de dólar. Nunca un presidente en ejercicio lo había hecho. Decisiones que se unen al largo catálogo de acciones con los que intenta pasar a la historia. Su cara ya aparece en algunos pases anuales para visitantes de los parques nacionales o en la llamada tarjeta dorada de visados, que permite a extranjeros millonarios vivir y trabajar tras pagar una gran suma de dinero. Su nombre también parece omnipresente. Figura en el edificio del Instituto de la Paz de Estados Unidos, o en la página web que vende medicamentos a precio reducido, la TrumpRx.gov, una página oficial del gobierno. Habrá una llamada clase Trump dentro de la nueva flota de buques de guerra que se están construyendo. Y en un supuesto despiste rebautizó el Estrecho de Hormuz como Estrecho de Trump. Lo que sí ha cambiado es el nombre de un histórico templo de la cultura estadounidense, el Kennedy Center. Ahora se llama el Trump Kennedy Center. Y esta mañana en Londres dos hombres judíos han sido acuchillados en plena calle. Sus heridas son graves, pero están estables. La policía ha detenido al supuesto atacante, que también ha intentado agredir a los agentes. El primer ministro del Reino Unido define el incidente como antisemita y profundamente preocupante. El premio Princesa de Asturias de las Artes es este año para la gran Patty Smith. El jurado ha destacado su creatividad impetuosa, pero también su compromiso frente a las injusticias y su inconformismo. Patti Smith lleva más de 50 años sobre los escenarios, regalándonos temas inolvidables y sigue llenando las salas de conciertos. La vemos con claridad, aunque solo la intuyamos mujer de blanco sobre fondo negro. Entra Patti Smith en el escenario del teatro real y con ella las siete vidas vividas, los amores perdidos, un Nueva York que ya no existe o el Hotel Chelsea. Entra con ella el activismo, el underground, el punk y la mística. Hija de una cantante de jazz y de un bailarín de claqué aficionado, fue testiga de Jehová. Vivió una infancia con menos de lo justo y una adolescencia marcada por un embarazo y la entrega en adopción de su hijo. Tuvo dos grandes amores, el fotógrafo Robert Mapplethorpe y el músico Fred Sonic Smith. Uno fue el artista de su vida, el otro el amor de esa vida. En 1975 publica un disco que es Una catedral, horror, con temas como Gloria. Su mayor éxito comercial es que cae este Because the Night, co-escrito con Bruce Smith. Dylan la adora, de hecho, ella cantó en la entrega del Nobel de Literatura. La escritora se retrata a sí misma en libros como Éramos unos niños o Pan de Ángeles. Octubre llegará a Asturias, a Madrid cada vez que viene acude a ver el Guernica y a tomar un bocadillo de calamares. El Niemeyer, la sidra y el cachopa guardan impacientes a la mujer de larga cabellera blanca  que tuvo claras dos cosas desde pequeña. Una, que nunca usaría pintalabio rojo. La otra, que el poder lo tiene la gente."
    ]
    kw = [
        "PP"
    ]

    searched_fragments = service.search_relevant_fragments(fragments, kw, { "Content-Type": "application/json" })
    print(searched_fragments)

    final_fragments = service.check_relevant_fragments(searched_fragments, kw, { "Content-Type": "application/json" })
    print(final_fragments)

