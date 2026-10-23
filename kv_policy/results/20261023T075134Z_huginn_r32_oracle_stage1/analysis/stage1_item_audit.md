# Stage 1.4 item audit of the R32 quality run

Source: `results/20261023T060125Z_huginn_r32_h200x8` (all generation records, 1200 rows).

| dataset | policy | n | acc | hit cap | cap & loop | cap & answer line | ended, no parse | scorer disagreements | loops (any) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | full | 200 | 0.415 | 3 | 3 | 0 | 0 | 16 | 3 |
| gsm8k | last | 200 | 0.445 | 1 | 1 | 0 | 0 | 18 | 1 |
| gsm8k | average | 200 | 0.420 | 2 | 2 | 0 | 0 | 9 | 2 |
| math500 | full | 200 | 0.155 | 47 | 42 | 0 | 1 | 3 | 53 |
| math500 | last | 200 | 0.165 | 52 | 50 | 0 | 1 | 3 | 60 |
| math500 | average | 200 | 0.125 | 49 | 48 | 0 | 1 | 2 | 55 |

Cap-hit generations that are NOT flagged as repetition loops (inspect manually):

- math500 average `test/intermediate_algebra/232.json` max16gram=1: …238), (0,239), (0,240), (0,241), (0,242), (0,243), (0,244), (0,245), (0,246), (0,247), (0,248), (0,2
- math500 full `test/intermediate_algebra/1279.json` max16gram=2: … 6x^4 + 11x^2 - 6 + 2x^6 - 12x^4 + 18x^2 - 12 + 3x^6 - 15x^4 + 21x^2 - 12 + 4x^6 - 20x^4 + 24x^2 - 1
- math500 full `test/number_theory/838.json` max16gram=1: …dot523\cdot524\cdot525\cdot526\cdot527\cdot528\cdot529\cdot530\cdot531\cdot532\cdot533\cdot534\cdot5
- math500 full `test/algebra/297.json` max16gram=2: …ational roots of the original equation are $\pm1, \pm2, \pm4, \pm5, \pm10, \pm20$. We can check that
- math500 full `test/number_theory/22.json` max16gram=1: …00,101201,101604,102009,102424,102841,103264,103709,104156,104625,105004,105401,105800,106201,106604
- math500 full `test/intermediate_algebra/232.json` max16gram=1: …6), (0,247), (0,248), (0,249), (0,250), (0,251), (0,252), (0,253), (0,254), (0,255), (0,256), (0,257
- math500 last `test/number_theory/838.json` max16gram=1: …dot523\cdot524\cdot525\cdot526\cdot527\cdot528\cdot529\cdot530\cdot531\cdot532\cdot533\cdot534\cdot5
- math500 last `test/number_theory/22.json` max16gram=1: …$23431$, $23536$, $23641$, $23746$, $23851$, $23956$, $24061$, $24166$, $24271$, $24376$, $24481$, $

Scorer disagreements (primary `correct` vs any reference metric):

- gsm8k average `gsm8k-test-0876` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='41' gold='41'
- gsm8k average `gsm8k-test-1004` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='8' gold='8'
- gsm8k average `gsm8k-test-0824` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='52' gold='52'
- gsm8k average `gsm8k-test-1233` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='81' gold='81'
- gsm8k average `gsm8k-test-0597` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='90' gold='90'
- gsm8k average `gsm8k-test-0223` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='20000' gold='20000'
- gsm8k average `gsm8k-test-0237` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='90' gold='90'
- gsm8k average `gsm8k-test-1091` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='1000' gold='1000'
- gsm8k average `gsm8k-test-0194` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='10' gold='10'
- gsm8k full `gsm8k-test-0575` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='36' gold='36'
- gsm8k full `gsm8k-test-0868` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='15' gold='15'
- gsm8k full `gsm8k-test-0631` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='4800' gold='4800'
- gsm8k full `gsm8k-test-1206` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='40000' gold='40000'
- gsm8k full `gsm8k-test-0258` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=False) pred='6' gold='6'
- gsm8k full `gsm8k-test-0852` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='123' gold='123'
- gsm8k full `gsm8k-test-1233` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='81' gold='81'
- gsm8k full `gsm8k-test-1191` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='100' gold='100'
- gsm8k full `gsm8k-test-1197` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='2400' gold='2400'
- gsm8k full `gsm8k-test-1091` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='1000' gold='1000'
- gsm8k full `gsm8k-test-0152` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='4' gold='4'
- gsm8k full `gsm8k-test-0872` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=False) pred='13' gold='13'
- gsm8k full `gsm8k-test-0302` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='15' gold='15'
- gsm8k full `gsm8k-test-0078` correct=False (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='15' gold='6'
- gsm8k full `gsm8k-test-0131` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='940' gold='940'
- gsm8k full `gsm8k-test-0799` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='4' gold='4'
- gsm8k last `gsm8k-test-0575` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='36' gold='36'
- gsm8k last `gsm8k-test-0868` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='15' gold='15'
- gsm8k last `gsm8k-test-0631` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='4800' gold='4800'
- gsm8k last `gsm8k-test-1206` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='40000' gold='40000'
- gsm8k last `gsm8k-test-0258` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=False) pred='6' gold='6'
- gsm8k last `gsm8k-test-0852` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='123' gold='123'
- gsm8k last `gsm8k-test-1233` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='81' gold='81'
- gsm8k last `gsm8k-test-1191` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='100' gold='100'
- gsm8k last `gsm8k-test-1197` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='2400' gold='2400'
- gsm8k last `gsm8k-test-1091` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='1000' gold='1000'
- gsm8k last `gsm8k-test-1312` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='180' gold='180'
- gsm8k last `gsm8k-test-0152` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='4' gold='4'
- gsm8k last `gsm8k-test-0872` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=False) pred='13' gold='13'
- gsm8k last `gsm8k-test-0302` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='15' gold='15'
- gsm8k last `gsm8k-test-0078` correct=False (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='15' gold='6'
- gsm8k last `gsm8k-test-0194` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='10' gold='10'
- gsm8k last `gsm8k-test-0131` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='940' gold='940'
- gsm8k last `gsm8k-test-0799` correct=True (lm_eval_strict_match=False, lm_eval_flexible_extract=True) pred='4' gold='4'
- math500 average `test/counting_and_probability/14.json` correct=True (math_verify=True, minerva_is_equiv=True, normalized_string_equal=False) pred='$\\frac{1}{4}$' gold='\\frac14'
- math500 average `test/intermediate_algebra/1714.json` correct=True (math_verify=True, minerva_is_equiv=False, normalized_string_equal=True) pred='(5, \\infty)' gold='(5,\\infty)'
- math500 full `test/counting_and_probability/14.json` correct=True (math_verify=True, minerva_is_equiv=True, normalized_string_equal=False) pred='$\\frac{1}{4}$' gold='\\frac14'
- math500 full `test/intermediate_algebra/1714.json` correct=True (math_verify=True, minerva_is_equiv=False, normalized_string_equal=True) pred='$(5,\\infty)$' gold='(5,\\infty)'
- math500 full `test/prealgebra/505.json` correct=True (math_verify=True, minerva_is_equiv=True, normalized_string_equal=False) pred='$x=6$' gold='6'
- math500 last `test/prealgebra/1646.json` correct=True (math_verify=True, minerva_is_equiv=True, normalized_string_equal=False) pred='80^\\circ' gold='80'
- math500 last `test/counting_and_probability/14.json` correct=True (math_verify=True, minerva_is_equiv=True, normalized_string_equal=False) pred='$\\frac{1}{4}$' gold='\\frac14'
- math500 last `test/intermediate_algebra/1714.json` correct=True (math_verify=True, minerva_is_equiv=False, normalized_string_equal=True) pred='$(5, \\infty)$' gold='(5,\\infty)'
