# Próximos pasos

Debido a que me he quedado sin tiempo, aquí añado un par de cambios que había pensado pero no tuve tiempo para implementar:

## Utilización de fuentes

Crear una clase abstracta  `Source` o `Adapter` que sea la base para integrar múltiples fuentes. Por ejemplo:

* YouTubeAdapter
* PodcastAdapter
* AudioAdapter
* TextFileAdapter
* RawTextAdapter
* PdfAdapter

Cada uno de estos adaptadores tendría un método `_parse` (recolector de información), `_process` (transformador de elemento en un texto) y `to_text` (método expuesto al exterior, encargado de llamar a los otros dos).

De esta forma, la recolecta de información se vuelve tan sencilla como:

```python
adapters: list[Adapter] = [...]

for i, adapter in enumerate(adapters):
    pct = i * pct_per_adapter
    print(f"PROGRESS:{pct}:Recogiendo información de adaptador de {adapter.source}.")
    
    text = adapter.to_text()
    if text: write_file(text)

    print("Información recogida correctamente.")
```

Asumo que esto quitaría al menos 100 líneas de código del archivo de pipeline y ayudaría bastante a su comprensión.
