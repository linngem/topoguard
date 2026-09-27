# Plan de análisis fijado antes de ejecutar Haiku (27-09-2026, 17:2x)

- Plausibilidad: media del panel (Sonnet 5 + Opus 5.5, 3 réplicas) en `plausibility.csv`
  (sha256 prefijo `ebf2f8e89b89eadc`), usada en escala log-odds.
- Microexperimento (Haiku 4.5, 12 casos, 4 réplicas): n ∈ {2,4,8}, m según `ADOPT_GRID`;
  retención con n = 4, m ∈ {0,2,4}.
- Pregunta 1 (proporción vs recuento): comparar modelos logísticos `fraccion`, `recuento`,
  `ambos` por log-loss con validación dejando fuera un caso. Gana el de menor log-loss.
- Pregunta 2 (plausibilidad): coeficiente de plausibilidad con IC95 % por bootstrap de casos.
- Pregunta 3 (red): con la regla ganadora del microexperimento, SIN ajustar nada con datos de
  red, predecir el alcance en los casos neumonía (datos ya existentes), iam_inferior y lupus.
  Métricas: Spearman y MAE por tipo de hallazgo. Comparar con la regla de fracción.
