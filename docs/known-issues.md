# Known Issues

## pytubefix

Existe un bug documentado en Pytubefix que provoca que tomar el parámetro `videos` de un objeto `Channel` devuelva una lista vacía. Para nuestro código, esto causa errores consistentes en el método `_filter_channel_videos` de `data_ingestion.download_yt_video`, impiendo recoger los vídeos de un canal.

El error es externo a nuestra implementación y depende de la resolución en la librería oficial, aquí documentada: <https://github.com/JuanBindez/pytubefix/issues/625>.
